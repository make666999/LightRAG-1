"""Tests for shared pytest harness helpers and process isolation guarantees.

Covers:
- ``.env`` isolation established in ``tests/__init__.py``
- ``lightrag_caplog`` / ``capture_lightrag_logs`` (``tests/_log_capture.py``)
- ``patch_module_sleep`` (``tests/_sleep_patch.py``)
"""

from __future__ import annotations

import logging
import os
import threading
import time

import pytest
from dotenv import find_dotenv, load_dotenv

from tests._log_capture import capture_lightrag_logs
from tests._sleep_patch import patch_module_sleep

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# .env isolation (tests/__init__.py)
# ---------------------------------------------------------------------------

_DOTENV_KEY = "LIGHTRAG_TEST_DOTENV_ISOLATION_PROBE"


@pytest.fixture
def dotenv_dir(tmp_path, monkeypatch):
    # Skip only for an explicit opt-out. A missing variable means the switch in
    # tests/__init__.py is gone, and the tests below must then fail.
    switch = os.environ.get("PYTHON_DOTENV_DISABLED")
    if switch is not None and switch.casefold() not in {"1", "true", "t", "yes", "y"}:
        pytest.skip("PYTHON_DOTENV_DISABLED explicitly turned off for this run")
    (tmp_path / ".env").write_text(f"{_DOTENV_KEY}=leaked\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_DOTENV_KEY, raising=False)
    return tmp_path


def test_load_dotenv_with_a_path_is_inert(dotenv_dir):
    """LightRAG's own pattern: ``load_dotenv(".env")`` at import time."""
    load_dotenv(dotenv_path=".env", override=False)
    assert _DOTENV_KEY not in os.environ


def test_a_searching_load_dotenv_is_inert(dotenv_dir):
    """A dependency's pattern: a bare search for the nearest ``.env``.

    pymilvus calls ``load_dotenv()`` on import, which walks up from its own
    install directory and found the checkout's ``.env`` whenever the venv sat
    inside it. An old python-dotenv ignores the switch; this test fails then.
    """
    load_dotenv(find_dotenv(usecwd=True), override=False)
    assert _DOTENV_KEY not in os.environ


# ---------------------------------------------------------------------------
# Log capture (tests/_log_capture.py)
# ---------------------------------------------------------------------------

_logger = logging.getLogger("lightrag")


def _warnings_with(caplog, needle: str) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


def test_fixture_captures_one_record_per_call(lightrag_caplog):
    with lightrag_caplog.at_level(logging.WARNING, logger="lightrag"):
        _logger.warning("fixture-probe")

    assert len(_warnings_with(lightrag_caplog, "fixture-probe")) == 1
    assert lightrag_caplog.text.count("fixture-probe") == 1


def test_context_manager_captures_one_record_per_call(caplog):
    with capture_lightrag_logs(caplog), caplog.at_level(logging.WARNING):
        _logger.warning("helper-probe")

    assert len(_warnings_with(caplog, "helper-probe")) == 1


def test_context_manager_leaves_pytests_own_attachment_in_place(caplog):
    attached_before = caplog.handler in _logger.handlers
    with capture_lightrag_logs(caplog):
        assert caplog.handler in _logger.handlers
    assert (caplog.handler in _logger.handlers) == attached_before


def test_records_survive_the_block(caplog):
    with capture_lightrag_logs(caplog), caplog.at_level(logging.WARNING):
        _logger.warning("after-probe")
    _logger.warning("outside-probe")  # may or may not be seen; not asserted

    assert len(_warnings_with(caplog, "after-probe")) == 1


# ---------------------------------------------------------------------------
# Scoped time.sleep patch (tests/_sleep_patch.py)
# ---------------------------------------------------------------------------


def _sleep_here(seconds):
    time.sleep(seconds)


def test_the_target_modules_sleep_is_mocked_and_recorded():
    started = time.monotonic()
    with patch_module_sleep(__name__) as sleep:
        _sleep_here(5)
    assert time.monotonic() - started < 1
    assert [call.args[0] for call in sleep.call_args_list] == [5]


def test_another_threads_sleep_stays_real_and_unrecorded():
    """The CI flake: a background loop (ragas' analytics thread) sleeping
    while a test mocked ``time.sleep`` process-wide spun on the mock and
    flooded its call list. Here it must keep sleeping for real."""
    stop = threading.Event()
    counter = [0]
    # The loop must live in ANOTHER module, as ragas' thread does: only
    # calls made from the patched module are mocked.
    namespace = {"__name__": "some_library._analytics", "time": time}
    exec(
        "def loop(stop, counter):\n"
        "    while not stop.is_set():\n"
        "        time.sleep(0.01)\n"
        "        counter[0] += 1\n",
        namespace,
    )

    worker = threading.Thread(
        target=namespace["loop"], args=(stop, counter), daemon=True
    )
    worker.start()
    try:
        with patch_module_sleep(__name__) as sleep:
            time.sleep(0.2)  # this module's call: recorded, returns at once
            stop.wait(0.2)  # let the background loop run under the patch
    finally:
        stop.set()
        worker.join(timeout=5)

    assert [call.args[0] for call in sleep.call_args_list] == [0.2]
    # ~20 real 10 ms sleeps fit in 0.2 s; a loop spinning on a mock does
    # hundreds of thousands.
    assert counter[0] < 200
