"""Offline tests for VLM image input normalization, cache-key invariants, and binding payloads."""

from __future__ import annotations

import base64
import hashlib
import struct
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.llm._vision_utils import (
    _JPEG_MAX_SCAN_STEPS,
    _dimensions_from_bytes,
    image_audit_metadata,
    image_cache_metadata,
    normalize_image_inputs,
    read_image_dimensions,
)
from lightrag.utils import (
    _serialize_cache_variant,
    compute_args_hash,
    get_llm_cache_identity,
    serialize_llm_cache_identity,
)

pytestmark = pytest.mark.offline

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8"
    b"\xcf\xc0\x00\x00\x00\x03\x00\x01\x5c\xcc\xd9\x9e\x00\x00\x00\x00"
    b"IEND\xaeB`\x82"
)
PNG_B = PNG_BYTES[:-12] + b"\x01" + PNG_BYTES[-11:]  # 1-byte tweak => different hash
# Header-only 32x16 PNG: dimensions come from IHDR, nothing decodes pixels.
PNG_32x16 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 32, 16)
_JPEG_FRAME_32x16 = b"\xff\xc0\x00\x0b\x08\x00\x10\x00\x20\x01\x01\x11\x00"


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


PNG_B64 = _b64(PNG_BYTES)


def _webp(chunk: bytes, payload: bytes) -> bytes:
    body = b"WEBP" + chunk + len(payload).to_bytes(4, "little") + payload
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _vlm_cache_hash(prompt: str, images: list[dict[str, Any]] | None) -> str:
    normalized = normalize_image_inputs(images) if images else []
    identity = get_llm_cache_identity({}, role="vlm")
    return compute_args_hash(
        prompt,
        "",
        "",
        serialize_llm_cache_identity(identity),
        _serialize_cache_variant({"type": "json_object"}),
        _serialize_cache_variant(image_cache_metadata(normalized)),
    )


def _make_fake_openai_client():
    fake_choice = MagicMock()
    fake_choice.message.content = "ok"
    fake_choice.message.reasoning_content = None
    fake_choice.finish_reason = "stop"
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)
    fake_client.close = AsyncMock()
    return fake_client


# ---------------------------------------------------------------------------
# 1. Image input normalization, MIME handling, dimensions, and metadata shapes
# ---------------------------------------------------------------------------


def test_normalize_raw_base64_and_empty_inputs():
    assert normalize_image_inputs(None) == []
    assert normalize_image_inputs([]) == []

    [img] = normalize_image_inputs([PNG_B64])
    assert img.index == 0
    assert img.mime_type == "image/png"
    assert img.raw_bytes == PNG_BYTES
    assert img.sha256 == hashlib.sha256(PNG_BYTES).hexdigest()
    assert img.source_id is None
    assert img.source_file is None


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        pytest.param(f"DaTa:Image/Png;BaSe64,{PNG_B64}", "image/png", id="url"),
        pytest.param(f"DATA:IMAGE/JPG;BASE64,{PNG_B64}", "image/jpeg", id="url-alias"),
        pytest.param(
            {"base64": PNG_B64, "mime_type": "Image/JPG"},
            "image/jpeg",
            id="dict-alias",
        ),
    ],
)
def test_declared_mime_type_is_used_and_canonicalized(item, expected):
    [image] = normalize_image_inputs([item])
    assert image.mime_type == expected
    assert image.raw_bytes == PNG_BYTES
    assert image.base64_str == PNG_B64


@pytest.mark.parametrize(
    ("items", "exc_type"),
    [
        pytest.param([f"data:image/png;base65,{PNG_B64}"], ValueError, id="bad-marker"),
        pytest.param(
            [f"data:image/png;ba\u017fe64,{PNG_B64}"],
            ValueError,
            id="non-ascii-marker",
        ),
        pytest.param(["this is not base64@@@!!"], ValueError, id="invalid-base64"),
        pytest.param([12345], TypeError, id="unsupported-element-type"),
        pytest.param([{"mime_type": "image/png"}], ValueError, id="dict-no-base64"),
    ],
)
def test_normalize_rejects_invalid_inputs(items, exc_type):
    with pytest.raises(exc_type):
        normalize_image_inputs(items)


@pytest.mark.parametrize(
    ("metadata_segments", "frame_marker"),
    [
        pytest.param(0, b"\xff\xc0", id="baseline-0-meta"),
        pytest.param(1, b"\xff\xc0", id="baseline-1-meta"),
        pytest.param(2, b"\xff\xc0", id="baseline-2-meta"),
        pytest.param(0, b"\xff\xc2", id="progressive"),
    ],
)
def test_jpeg_dimensions_after_metadata(tmp_path, metadata_segments, frame_marker):
    comment = b"\xff\xfe\xff\xff" + b"x" * 65533
    frame = frame_marker + b"\x00\x0b\x08\x00\x10\x00\x20\x01\x01\x11\x00"
    raw = b"\xff\xd8" + comment * metadata_segments + frame + b"\xff\xd9"
    path = tmp_path / "image.jpg"
    path.write_bytes(raw)

    assert read_image_dimensions(path) == (32, 16)
    [image] = normalize_image_inputs([_b64(raw)])
    assert (image.width, image.height) == (32, 16)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\xff\xd8\xff",
        b"\xff\xd8\xff\xe1\x00",
        b"\xff\xd8\xff\xe1\x00\x01",
        b"\xff\xd8\xff\xe1\x00\x10abc",
        b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x10\x00\x20",
    ],
)
def test_incomplete_jpeg_dimensions_are_unknown(tmp_path, raw):
    path = tmp_path / "incomplete.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) is None


@pytest.mark.parametrize("end_marker", [b"\xff\xd9", b"\xff\xda\x00\x02"])
def test_jpeg_dimensions_stop_before_image_data(tmp_path, end_marker):
    raw = b"\xff\xd8" + end_marker + _JPEG_FRAME_32x16
    path = tmp_path / "no-frame.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) is None
    [image] = normalize_image_inputs([_b64(raw)])
    assert (image.width, image.height) == (None, None)


@pytest.mark.parametrize(
    "padding",
    [
        pytest.param(b"\xff\xe1\x00\x02" * (_JPEG_MAX_SCAN_STEPS + 1), id="segments"),
        pytest.param(b"\xff" * (_JPEG_MAX_SCAN_STEPS + 1), id="fill-bytes"),
    ],
)
def test_jpeg_scan_is_bounded(tmp_path, padding):
    raw = b"\xff\xd8" + padding + _JPEG_FRAME_32x16 + b"\xff\xd9"
    path = tmp_path / "crafted.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) is None
    assert _dimensions_from_bytes(raw) is None


def test_jpeg_scan_budget_admits_many_real_segments(tmp_path):
    segment = b"\xff\xe2\x10\x00" + b"i" * 0x0FFE
    raw = b"\xff\xd8" + segment * 512 + _JPEG_FRAME_32x16 + b"\xff\xd9"
    path = tmp_path / "many-segments.jpg"
    path.write_bytes(raw)
    assert read_image_dimensions(path) == (32, 16)


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_jpeg_dimensions_accept_bytes_like(wrap):
    raw = b"\xff\xd8" + _JPEG_FRAME_32x16 + b"\xff\xd9"
    assert _dimensions_from_bytes(wrap(raw)) == (32, 16)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(PNG_32x16, (32, 16), id="png"),
        pytest.param(b"GIF89a\x20\x00\x10\x00" + b"\x00" * 8, (32, 16), id="gif"),
        pytest.param(
            _webp(b"VP8 ", b"\x00" * 3 + b"\x9d\x01\x2a\x20\x00\x10\x00"),
            (32, 16),
            id="webp-vp8",
        ),
        pytest.param(
            _webp(b"VP8L", b"\x2f\x1f\xc0\x03\x00"),
            (32, 16),
            id="webp-vp8l",
        ),
        pytest.param(
            _webp(b"VP8X", b"\x00" * 4 + b"\x1f\x00\x00\x0f\x00\x00"),
            (32, 16),
            id="webp-vp8x",
        ),
    ],
)
def test_non_jpeg_dimensions_from_file_header(tmp_path, raw, expected):
    path = tmp_path / "image.bin"
    path.write_bytes(raw + b"\x00" * 64)
    assert read_image_dimensions(path) == expected


def test_image_cache_and_audit_metadata_shapes():
    images = normalize_image_inputs(
        [
            {
                "base64": _b64(PNG_32x16),
                "source_id": "img-001",
                "source_file": "/tmp/foo.png",
                "modality": "image",
                "doc_id": "doc-1",
            }
        ]
    )
    digest = hashlib.sha256(PNG_32x16).hexdigest()

    [cache_meta] = image_cache_metadata(images)
    assert cache_meta == {
        "index": 0,
        "mime_type": "image/png",
        "sha256": digest,
        "bytes": len(PNG_32x16),
        "width": 32,
        "height": 16,
    }

    [audit_meta] = image_audit_metadata(images)
    assert audit_meta == {
        **cache_meta,
        "source_id": "img-001",
        "source_file": "/tmp/foo.png",
        "modality": "image",
        "doc_id": "doc-1",
    }


# ---------------------------------------------------------------------------
# 2. VLM cache-key invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "expect_equal"),
    [
        pytest.param(
            ("describe", [{"base64": PNG_B64}]),
            ("describe", [{"base64": PNG_B64}]),
            True,
            id="same-prompt-same-image",
        ),
        pytest.param(
            ("describe", [{"base64": PNG_B64}]),
            ("describe", [{"base64": _b64(PNG_B)}]),
            False,
            id="same-prompt-different-image",
        ),
        pytest.param(
            (
                "describe",
                [
                    {
                        "base64": PNG_B64,
                        "source_id": "img-001",
                        "source_file": "/path/a/img.png",
                        "modality": "image",
                        "doc_id": "doc-1",
                    }
                ],
            ),
            (
                "describe",
                [
                    {
                        "base64": PNG_B64,
                        "source_id": "img-002",
                        "source_file": "/different/elsewhere/copy.png",
                        "modality": "image",
                        "doc_id": "doc-2",
                    }
                ],
            ),
            True,
            id="same-image-different-provenance",
        ),
        pytest.param(
            ("describe", [{"base64": PNG_B64}]),
            ("describe in english", [{"base64": PNG_B64}]),
            False,
            id="different-prompt-same-image",
        ),
        pytest.param(
            ("describe", None),
            ("describe", [{"base64": PNG_B64}]),
            False,
            id="image-present-vs-absent",
        ),
    ],
)
def test_vlm_cache_key_invariants(first, second, expect_equal):
    assert (_vlm_cache_hash(*first) == _vlm_cache_hash(*second)) is expect_equal


# ---------------------------------------------------------------------------
# 3. Per-binding image_inputs payload shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_input",
    [
        PNG_B64,
        f"data:image/png;BASE64,{PNG_B64}",
        f"DATA:image/png;base64,{PNG_B64}",
        f"DaTa:image/png;BaSe64,{PNG_B64}",
        f"DATA:IMAGE/PNG;BASE64,{PNG_B64}",
    ],
)
@pytest.mark.asyncio
async def test_openai_binding_inserts_image_url_content_block(image_input):
    from lightrag.llm import openai as openai_mod

    fake_client = _make_fake_openai_client()
    with patch.object(
        openai_mod, "create_openai_async_client", return_value=fake_client
    ):
        await openai_mod.openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="describe",
            api_key="dummy",
            image_inputs=[image_input],
        )

    _, kwargs = fake_client.chat.completions.create.call_args
    messages = kwargs["messages"]
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0] == {"type": "text", "text": "describe"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == f"data:image/png;base64,{PNG_B64}"


@pytest.mark.asyncio
async def test_openai_binding_text_only_remains_plain_string():
    from lightrag.llm import openai as openai_mod

    fake_client = _make_fake_openai_client()
    with patch.object(
        openai_mod, "create_openai_async_client", return_value=fake_client
    ):
        await openai_mod.openai_complete_if_cache(
            model="gpt-4o-mini",
            prompt="describe",
            api_key="dummy",
        )

    _, kwargs = fake_client.chat.completions.create.call_args
    assert kwargs["messages"][-1]["content"] == "describe"


@pytest.mark.asyncio
async def test_ollama_binding_attaches_images_to_user_message():
    from lightrag.llm import ollama as ollama_mod

    fake_client = MagicMock()
    fake_client.chat = AsyncMock(return_value={"message": {"content": "ok"}})
    fake_client._client = MagicMock()
    fake_client._client.aclose = AsyncMock()

    with patch.object(ollama_mod.ollama, "AsyncClient", return_value=fake_client):
        await ollama_mod._ollama_model_if_cache(
            model="llava",
            prompt="describe",
            image_inputs=[PNG_B64],
        )

    _, kwargs = fake_client.chat.call_args
    user_msg = kwargs["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "describe"
    assert user_msg["images"] == [PNG_B64]


@pytest.mark.asyncio
async def test_anthropic_binding_inserts_image_content_block():
    from lightrag.llm import anthropic as anthropic_mod

    captured: dict[str, Any] = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return MagicMock(content=[MagicMock(type="text", text="")])

    fake_client = MagicMock()
    fake_client.messages = FakeMessages()

    with patch.object(anthropic_mod, "AsyncAnthropic", return_value=fake_client):
        await anthropic_mod.anthropic_complete_if_cache(
            model="claude-3-opus",
            prompt="describe",
            api_key="dummy",
            image_inputs=[PNG_B64],
        )

    user_content = captured["messages"][-1]["content"]
    assert isinstance(user_content, list)
    image_blocks = [b for b in user_content if b.get("type") == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": PNG_B64,
    }
    assert user_content[-1] == {"type": "text", "text": "describe"}


@pytest.mark.asyncio
async def test_lollms_binding_rejects_image_inputs():
    from lightrag.llm import lollms as lollms_mod

    with pytest.raises(NotImplementedError):
        await lollms_mod.lollms_model_if_cache(
            model="unused",
            prompt="hi",
            image_inputs=[PNG_B64],
        )


@pytest.mark.asyncio
async def test_bedrock_binding_forces_non_stream_when_image_present():
    from lightrag.llm import bedrock as bedrock_mod

    captured: dict[str, Any] = {}

    class FakeBedrockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def converse(self, **kwargs):
            captured["mode"] = "converse"
            captured["args"] = kwargs
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "stopReason": "end_turn",
            }

        async def converse_stream(self, **kwargs):
            captured["mode"] = "converse_stream"
            captured["args"] = kwargs
            return {"stream": []}

    class FakeSession:
        def client(self, *_, **__):
            return FakeBedrockClient()

    with patch.object(bedrock_mod.aioboto3, "Session", return_value=FakeSession()):
        await bedrock_mod.bedrock_complete_if_cache(
            "anthropic.claude-3-haiku-20240307-v1:0",
            "describe",
            stream=True,
            image_inputs=[PNG_B64],
            aws_region="us-east-1",
        )

    assert captured["mode"] == "converse"
    user_msg = captured["args"]["messages"][-1]
    image_blocks = [block for block in user_msg["content"] if "image" in block]
    assert len(image_blocks) == 1
    assert image_blocks[0]["image"]["format"] == "png"
    assert image_blocks[0]["image"]["source"]["bytes"] == PNG_BYTES
