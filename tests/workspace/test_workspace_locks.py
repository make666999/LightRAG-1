"""Tests for workspace filesystem locks: anchor lock and working-dir snapshot claim.

Covers:
1. Anchor lock (``lightrag/kg/anchor_lock.py``): shared for starters, exclusive
   for offline migration, counted per process tree, path-normalized, and
   fail-open/fail-closed behavior when locking or writing is unavailable.
2. Working-dir lock (``lightrag/kg/working_dir_lock.py``): one process tree at
   a time on a file-backed configuration directory, cross-process exclusion via
   ``_working_dir_lock_probe.py``, kernel release on process death, refcounting,
   and symlink/path normalization.
"""

from __future__ import annotations

import builtins
import errno
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from lightrag.config_shards import json_config_dir
from lightrag.exceptions import ConfigurationAnchorLockError
from lightrag.kg import anchor_lock as al
from lightrag.kg import working_dir_lock as wdl
from lightrag.kg.working_dir_lock import (
    LOCK_FILENAME,
    acquire_working_dir_lock,
    holds_working_dir_lock,
    release_working_dir_lock,
    uses_working_dir,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _no_locks_leak():
    yield
    for hold in list(al._shared_holds.values()):
        if hold.handle is not None:
            try:
                hold.handle.close()
            except OSError:
                pass
    al._shared_holds.clear()
    for claim in list(wdl._claims.values()):
        try:
            if claim.handle is not None:
                claim.handle.close()
        except OSError:
            pass
    wdl._claims.clear()


# ---------------------------------------------------------------------------
# Anchor lock (Unix fcntl)
# ---------------------------------------------------------------------------


def _foreign(tmp_path, mode):
    """A lock taken the way ANOTHER process would hold it."""
    path = al.anchor_lock_path(str(tmp_path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a+")
    fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
    return handle


def _cannot_lock(monkeypatch):
    real = fcntl.flock

    def _flock(fd, op):
        if op & fcntl.LOCK_UN:
            return real(fd, op)
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", _flock)


@pytest.mark.skipif(fcntl is None, reason="anchor lock tests require fcntl")
class TestAnchorLock:
    def test_the_lock_file_is_distinct_from_the_snapshot_claim(self, tmp_path):
        """The anchor lock sits directly under ``working_dir`` -- the same
        directory as the empty workspace's JSON snapshot claim -- under its own
        name, so the two never collide."""
        path = al.anchor_lock_path(str(tmp_path))
        assert os.path.basename(path) == ".lightrag_anchor.lock"
        assert os.path.basename(path) != LOCK_FILENAME
        assert os.path.dirname(path) == os.path.realpath(tmp_path)
        assert path != os.path.join(json_config_dir(str(tmp_path), ""), LOCK_FILENAME)
        bind = al.anchor_bind_lock_path(str(tmp_path))
        assert os.path.basename(bind) == ".lightrag_anchor_bind.lock"
        assert os.path.dirname(bind) == os.path.dirname(path)

    def test_starters_share_it(self, tmp_path):
        other = _foreign(tmp_path, fcntl.LOCK_SH)
        try:
            al.acquire_anchor_lock_shared(str(tmp_path))
            assert al.holds_anchor_lock(str(tmp_path))
        finally:
            other.close()
        al.release_anchor_lock_shared(str(tmp_path))
        assert not al.holds_anchor_lock(str(tmp_path))

    def test_a_starter_is_refused_while_a_migration_holds_it(self, tmp_path):
        migration = _foreign(tmp_path, fcntl.LOCK_EX)
        try:
            with pytest.raises(ConfigurationAnchorLockError, match="migration"):
                al.acquire_anchor_lock_shared(str(tmp_path))
            assert not al.holds_anchor_lock(str(tmp_path))
        finally:
            migration.close()
        al.acquire_anchor_lock_shared(str(tmp_path))
        al.release_anchor_lock_shared(str(tmp_path))

    def test_the_migration_is_refused_while_another_starter_holds_it(self, tmp_path):
        starter = _foreign(tmp_path, fcntl.LOCK_SH)
        try:
            with pytest.raises(ConfigurationAnchorLockError, match="held by a running"):
                al.acquire_anchor_lock_exclusive(str(tmp_path), assume_exclusive=True)
        finally:
            starter.close()

    def test_the_migration_is_refused_while_this_process_is_a_starter(self, tmp_path):
        al.acquire_anchor_lock_shared(str(tmp_path))
        with pytest.raises(ConfigurationAnchorLockError, match="already holds"):
            al.acquire_anchor_lock_exclusive(str(tmp_path))
        al.release_anchor_lock_shared(str(tmp_path))

    def test_the_migration_excludes_every_starter_and_releases(self, tmp_path):
        lock = al.acquire_anchor_lock_exclusive(str(tmp_path))
        assert lock.enforced is True
        with pytest.raises(BlockingIOError):
            _foreign(tmp_path, fcntl.LOCK_SH)
        lock.release()
        lock.release()  # idempotent
        _foreign(tmp_path, fcntl.LOCK_SH).close()

    def test_holds_are_counted_per_process_tree(self, tmp_path):
        """Two instances (or a master and its worker) count into one hold; the
        lock is given back only at zero."""
        al.acquire_anchor_lock_shared(str(tmp_path))
        al.acquire_anchor_lock_shared(str(tmp_path))
        al.release_anchor_lock_shared(str(tmp_path))
        assert al.holds_anchor_lock(str(tmp_path))
        with pytest.raises(BlockingIOError):
            _foreign(tmp_path, fcntl.LOCK_EX)
        al.release_anchor_lock_shared(str(tmp_path))
        assert not al.holds_anchor_lock(str(tmp_path))
        _foreign(tmp_path, fcntl.LOCK_EX).close()

    def test_releasing_without_a_hold_is_a_no_op(self, tmp_path):
        al.release_anchor_lock_shared(str(tmp_path))
        assert not al.holds_anchor_lock(str(tmp_path))

    def test_two_spellings_of_one_directory_are_one_hold(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path.parent)
        al.acquire_anchor_lock_shared(str(tmp_path))
        al.acquire_anchor_lock_shared(tmp_path.name)
        assert len(al._shared_holds) == 1
        al.release_anchor_lock_shared(tmp_path.name)
        al.release_anchor_lock_shared(str(tmp_path))
        assert not al._shared_holds

    def test_a_filesystem_that_cannot_lock_lets_starters_proceed(
        self, tmp_path, monkeypatch
    ):
        _cannot_lock(monkeypatch)
        warnings: list[str] = []
        monkeypatch.setattr(al.logger, "warning", warnings.append)
        al.acquire_anchor_lock_shared(str(tmp_path))
        assert al.holds_anchor_lock(str(tmp_path))
        assert any("proceeding without it" in w for w in warnings)
        al.release_anchor_lock_shared(str(tmp_path))

    def test_a_filesystem_that_cannot_lock_refuses_the_migration(
        self, tmp_path, monkeypatch
    ):
        _cannot_lock(monkeypatch)
        with pytest.raises(ConfigurationAnchorLockError, match="--assume-exclusive"):
            al.acquire_anchor_lock_exclusive(str(tmp_path))
        lock = al.acquire_anchor_lock_exclusive(str(tmp_path), assume_exclusive=True)
        assert lock.enforced is False
        lock.release()

    def test_a_platform_without_shared_locks_fails_open_and_refuses_the_migration(
        self, tmp_path, monkeypatch
    ):
        """``msvcrt.locking`` has no shared mode, so on Windows starters cannot
        announce themselves -- and an exclusive lock there would exclude nobody."""
        monkeypatch.setattr(al, "_fcntl", lambda: None)
        al.acquire_anchor_lock_shared(str(tmp_path))
        assert al._shared_holds[al.anchor_lock_path(str(tmp_path))].enforced is False
        al.release_anchor_lock_shared(str(tmp_path))
        with pytest.raises(ConfigurationAnchorLockError):
            al.acquire_anchor_lock_exclusive(str(tmp_path))
        al.acquire_anchor_lock_exclusive(str(tmp_path), assume_exclusive=True).release()

    def test_a_read_only_directory_lets_starters_proceed(self, tmp_path, monkeypatch):
        def _read_only(path):
            raise OSError(errno.EROFS, "Read-only file system", path)

        monkeypatch.setattr(al, "_open_lock_file", _read_only)
        al.acquire_anchor_lock_shared(str(tmp_path))
        assert al.holds_anchor_lock(str(tmp_path))
        al.release_anchor_lock_shared(str(tmp_path))
        with pytest.raises(ConfigurationAnchorLockError):
            al.acquire_anchor_lock_exclusive(str(tmp_path))


# ---------------------------------------------------------------------------
# Working directory lock (cross-process snapshot directory claim)
# ---------------------------------------------------------------------------

_PROBE = Path(__file__).with_name("_working_dir_lock_probe.py")
# The probe runs as a script, so its sys.path starts at its own directory:
# without the root on PYTHONPATH it imports whichever lightrag is installed,
# which in a second checkout is not the code under test.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_PROBE_ENV = {
    **os.environ,
    "PYTHONPATH": os.pathsep.join(
        p for p in (_REPO_ROOT, os.environ.get("PYTHONPATH")) if p
    ),
}


def _probe(command: str, path) -> subprocess.CompletedProcess:
    """Run one probe command in a fresh interpreter -- a process tree that did
    NOT inherit this one's bookkeeping or its lock descriptor."""
    result = subprocess.run(
        [sys.executable, str(_PROBE), command, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        env=_PROBE_ENV,
    )
    assert result.returncode == 0, (
        f"probe '{command}' failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


class _ForeignTree:
    """The ``serve`` probe: another process tree that answers claim attempts."""

    def __init__(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(_PROBE), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_PROBE_ENV,
        )
        self._answers: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            self._answers.put(line.strip())
        self._answers.put("")  # the probe exited

    def attempt(self, path) -> str:
        self._proc.stdin.write(f"{path}\n")
        self._proc.stdin.flush()
        try:
            answer = self._answers.get(timeout=60)
        except queue.Empty:
            pytest.fail("the foreign probe did not answer within 60s")
        if not answer:
            self._proc.wait(timeout=10)
            pytest.fail(f"the foreign probe exited: {self._proc.stderr.read()}")
        return answer

    def close(self) -> None:
        self._proc.stdin.close()
        rc = self._proc.wait(timeout=30)
        assert rc == 0, self._proc.stderr.read()


@pytest.fixture(scope="module")
def foreign_tree():
    tree = _ForeignTree()
    yield tree
    tree.close()


@pytest.fixture
def _foreign_attempt(foreign_tree):
    return foreign_tree.attempt


def test_only_a_file_backed_configuration_storage_claims_the_directory():
    """A server-backed configuration keeps its rows on the server, so two
    servers there share a container by design and nothing is claimed."""
    assert uses_working_dir("JsonKVStorage") is True
    assert uses_working_dir("RedisKVStorage") is False
    assert uses_working_dir("PGKVStorage") is False
    assert uses_working_dir(None) is False


def test_a_second_process_tree_is_refused(tmp_path, _foreign_attempt):
    acquire_working_dir_lock(str(tmp_path))
    assert _foreign_attempt(tmp_path) == "REFUSED"


def test_the_directory_is_free_once_the_holder_releases(tmp_path, _foreign_attempt):
    acquire_working_dir_lock(str(tmp_path))
    release_working_dir_lock(str(tmp_path))

    assert holds_working_dir_lock(str(tmp_path)) is False
    assert _foreign_attempt(tmp_path) == "ADMITTED"


def test_a_holder_that_dies_leaves_nothing_to_reap(tmp_path):
    """Why an OS lock and not a PID file: no stale-file cleanup, no
    read-the-PID-then-probe-liveness race. The kernel releases it."""
    _probe("dying", tmp_path)

    # The lock file still exists; that is not what holds the lock.
    assert (tmp_path / LOCK_FILENAME).exists()
    acquire_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is True


def test_a_second_instance_in_one_tree_counts_itself_in(tmp_path, _foreign_attempt):
    """Several LightRAG instances in one process are legal, so the claim is
    counted and released only by the last one."""
    acquire_working_dir_lock(str(tmp_path))
    acquire_working_dir_lock(str(tmp_path))

    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is True
    assert _foreign_attempt(tmp_path) == "REFUSED"

    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_a_filesystem_that_cannot_lock_warns_and_proceeds(tmp_path, monkeypatch):
    """NFSv3 without lockd, SMB/CIFS. Refusing there would break deployments
    that work, to protect against a rarer failure -- so it fails OPEN."""
    monkeypatch.setattr(
        wdl, "_try_lock", lambda handle: (_ for _ in ()).throw(OSError("no locks"))
    )

    acquire_working_dir_lock(str(tmp_path))

    assert holds_working_dir_lock(str(tmp_path)) is True
    assert wdl._claims[wdl._lock_path(str(tmp_path))].enforced is False


def test_a_lock_file_that_cannot_be_opened_warns_and_proceeds(
    tmp_path, monkeypatch, lightrag_log_records
):
    """A lock file left by a run under another user, or a read-only
    directory: the storages that worked before the claim existed must still
    start, so it fails OPEN like a filesystem that cannot lock."""
    real_open = builtins.open

    def _refuse(path, *args, **kwargs):
        if str(path).endswith(LOCK_FILENAME):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _refuse)

    acquire_working_dir_lock(str(tmp_path))

    assert holds_working_dir_lock(str(tmp_path)) is True
    assert wdl._claims[wdl._lock_path(str(tmp_path))].enforced is False
    assert any(
        "Could not open the lock file" in record.getMessage()
        for record in lightrag_log_records
    )
    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_releasing_without_a_claim_is_a_no_op(tmp_path):
    release_working_dir_lock(str(tmp_path))
    assert holds_working_dir_lock(str(tmp_path)) is False


def test_two_spellings_of_one_directory_are_one_claim(tmp_path, _foreign_attempt):
    """``_lock_path`` resolves symlinks, and that is load bearing rather than
    tidiness: a relative path and its absolute form, or a symlink and its
    target, must produce ONE key. Two keys mean this tree opens a second
    descriptor on the one file -- and ``flock`` belongs to the open file
    description, so the process refuses ITSELF.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    acquire_working_dir_lock(str(real))
    acquire_working_dir_lock(str(link))  # counts in; does not refuse itself

    assert holds_working_dir_lock(str(link)) is True
    release_working_dir_lock(str(link))
    assert _foreign_attempt(real) == "REFUSED"

    release_working_dir_lock(str(real))
    assert _foreign_attempt(real) == "ADMITTED"
