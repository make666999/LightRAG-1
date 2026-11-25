"""Regression tests for external parser download-safety bounds (Docling + MinerU).

Both ``DoclingRawClient._download_result_into`` and ``MinerURawClient._download_zip``
share the download safety helpers in ``lightrag.parser.external._common``:
- an overall wall-clock deadline around the download, independent of the
  per-read httpx timeout;
- a streaming byte cap that rejects an oversized response mid-download
  instead of buffering it in full first;
- defense-in-depth against zip declared-size lies and non-2xx error body
  survival on genuinely unread byte streams.

Calls each client's download method directly against a real ``httpx.AsyncClient``
backed by ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import httpx
import pytest

from lightrag.parser.external.docling.client import DoclingRawClient
from lightrag.parser.external.mineru.client import MinerURawClient

pytestmark = pytest.mark.offline


def _small_valid_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content_list.json", "[]")
    return buf.getvalue()


@dataclass(frozen=True)
class _DownloadAdapter:
    """Engine-specific download entrypoint + expected HTTP 500 error prefix."""

    engine: str
    download: Callable[[httpx.AsyncClient, Path], Awaitable[None]]
    http_500_prefix: str
    zip_headers: dict[str, str]


@pytest.fixture(params=["docling", "mineru"])
def adapter(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> _DownloadAdapter:
    if request.param == "docling":
        monkeypatch.setenv("DOCLING_ENDPOINT", "http://docling.test")

        async def _download(client: httpx.AsyncClient, raw_dir: Path) -> None:
            await DoclingRawClient()._download_result_into(
                client, "task-1", raw_dir, "demo.pdf"
            )

        return _DownloadAdapter(
            engine="docling",
            download=_download,
            http_500_prefix="Docling result task-1 download failed: HTTP 500",
            zip_headers={"content-type": "application/zip"},
        )

    monkeypatch.setenv("MINERU_API_MODE", "official")
    monkeypatch.setenv("MINERU_API_TOKEN", "token-123")

    async def _download(client: httpx.AsyncClient, raw_dir: Path) -> None:
        await MinerURawClient()._download_zip(client, "https://x/result.zip", raw_dir)

    return _DownloadAdapter(
        engine="mineru",
        download=_download,
        http_500_prefix="MinerU result bundle download failed: HTTP 500",
        zip_headers={},
    )


@pytest.mark.asyncio
async def test_download_deadline_cuts_off_a_slow_peer(
    adapter: _DownloadAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A response that never arrives within the configured deadline must be
    rejected well before httpx's own (much larger) per-read timeout."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(
            200, content=_small_valid_zip(), headers=adapter.zip_headers
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "0.05")
    transport = httpx.MockTransport(slow_handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    start = time.monotonic()
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="wall-clock deadline"):
            await adapter.download(client, raw_dir)
    elapsed = time.monotonic() - start

    assert elapsed < 0.2, (
        f"expected the deadline to cut the call off well before the "
        f"mocked 0.3s response, took {elapsed:.3f}s instead"
    )


@pytest.mark.asyncio
async def test_download_disabled_deadline_lets_a_slow_peer_finish(
    adapter: _DownloadAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control case: a non-positive deadline disables the gate, matching the
    zip-bomb guards' 'non-positive = unlimited' convention."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(
            200, content=_small_valid_zip(), headers=adapter.zip_headers
        )

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "0")
    transport = httpx.MockTransport(slow_handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async with httpx.AsyncClient(transport=transport) as client:
        await adapter.download(client, raw_dir)

    assert (raw_dir / "content_list.json").is_file()


@pytest.mark.asyncio
async def test_download_external_cancellation_is_not_a_timeout(
    adapter: _DownloadAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``asyncio.timeout()`` must distinguish its own deadline firing from an
    unrelated external cancellation. Proves all five contract points:
    CancelledError escapes; not converted to TimeoutError/RuntimeError;
    no archive content extracted; response stream closes; client closes."""
    captured: dict = {}

    async def hanging_body():
        yield b"partial-chunk-before-the-stall"
        await asyncio.Event().wait()  # never set: stalls forever
        yield b"unreachable"  # pragma: no cover

    async def hanging_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=hanging_body(), headers=adapter.zip_headers)

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_TIMEOUT", "0")
    transport = httpx.MockTransport(hanging_handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    client = httpx.AsyncClient(transport=transport)
    real_stream = client.stream

    class _CapturingStreamCM:
        """Wraps httpx's stream context manager to capture the entered response."""

        def __init__(self, inner):
            self._inner = inner

        async def __aenter__(self):
            response = await self._inner.__aenter__()
            captured["response"] = response
            return response

        async def __aexit__(self, *exc):
            return await self._inner.__aexit__(*exc)

    def _spy_stream(*args, **kwargs):
        return _CapturingStreamCM(real_stream(*args, **kwargs))

    client.stream = _spy_stream  # type: ignore[method-assign]

    async def run() -> None:
        async with client:
            await adapter.download(client, raw_dir)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)  # let it receive headers + first chunk, then stall
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert exc_info.type is asyncio.CancelledError
    assert not (raw_dir / "content_list.json").exists()
    response = captured.get("response")
    assert response is not None, "expected the stream to have been entered"
    assert response.is_closed
    assert client.is_closed


@pytest.mark.asyncio
async def test_download_rejects_oversized_response_before_full_buffer(
    adapter: _DownloadAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A response larger than the configured budget is rejected by the
    streaming cap, not merely by safe_extract_zip after full buffering."""
    oversized_zip = _small_valid_zip() + b"\x00" * 10_000

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=oversized_zip, headers=adapter.zip_headers)

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES", "100")
    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="exceeds 100 bytes"):
            await adapter.download(client, raw_dir)


@pytest.mark.asyncio
async def test_download_wire_cap_is_independent_of_uncompressed_cap(
    adapter: _DownloadAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES`` (uncompressed) and
    ``PARSER_RESULT_BUNDLE_DOWNLOAD_MAX_BYTES`` (wire bytes) are independent:
    a tiny uncompressed cap must fail at ``safe_extract_zip``, not streaming."""
    small_zip = _small_valid_zip()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=small_zip, headers=adapter.zip_headers)

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES", "1")
    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="uncompressed size"):
            await adapter.download(client, raw_dir)


@pytest.mark.asyncio
async def test_download_declared_size_lie_still_caught_when_raw_is_small(
    adapter: _DownloadAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: a small-on-the-wire zip that lies about its own
    declared uncompressed size must still be caught by ``safe_extract_zip``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("content_list.json", "[]")
        info = zf.infolist()[-1]
        info.file_size = 10 * 1024 * 1024 * 1024  # lies: declares 10 GiB
    lying_zip = buf.getvalue()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=lying_zip, headers=adapter.zip_headers)

    monkeypatch.setenv("PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES", str(1024 * 1024))
    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError, match="uncompressed size"):
            await adapter.download(client, raw_dir)


class _GenuineStream(httpx.AsyncByteStream):
    """A real streamed byte source with nothing preloaded.

    ``httpx.Response(..., content=b"...")`` marks the response already-read at
    construction time and would never catch a regression in how
    ``stream_capped_get`` threads collected bytes into
    ``raise_for_status_with_detail``.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "chunks", "expected_detail"),
    [
        (
            "text/plain",
            [b"disk full", b": no space for temp file"],
            "disk full: no space for temp file",
        ),
        (
            "application/json",
            [b'{"error": "', b'disk full"}'],
            '{"error": "disk full"}',
        ),
    ],
    ids=["text-body", "json-body"],
)
async def test_download_http_error_detail_survives_genuine_streaming(
    adapter: _DownloadAdapter,
    tmp_path: Path,
    content_type: str,
    chunks: list[bytes],
    expected_detail: str,
) -> None:
    """A non-2xx response's body text must reach the raised ``RuntimeError``
    (exact type, not ``httpx.ResponseNotRead``) after manual streaming."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            stream=_GenuineStream(chunks),
            headers={"content-type": content_type},
        )

    transport = httpx.MockTransport(handler)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(RuntimeError) as excinfo:
            await adapter.download(client, raw_dir)

    assert type(excinfo.value) is RuntimeError, (
        f"expected our own RuntimeError, got {type(excinfo.value).__name__} "
        f"(httpx.ResponseNotRead escaping disguised as RuntimeError would "
        f"also pass a bare pytest.raises(RuntimeError) check)"
    )
    message = str(excinfo.value)
    assert adapter.http_500_prefix in message
    assert expected_detail in message
    assert not (raw_dir / "content_list.json").exists()
