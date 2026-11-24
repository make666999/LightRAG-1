"""Tests for ``lightrag-download-cache`` (spaCy model install/download and tiktoken cache path)."""

import os
import subprocess
import sys

import pytest

import lightrag.tools.download_cache as download_cache

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# spaCy model installation and wheel download
# --------------------------------------------------------------------------- #


@pytest.fixture
def recorded_runs(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _set_tools(monkeypatch, *, pip: bool, uv: str | None):
    import importlib.util
    import shutil

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (
            (object() if pip else None)
            if name == "pip"
            else real_find_spec(name, *a, **k)
        ),
    )
    monkeypatch.setattr(shutil, "which", lambda name: uv if name == "uv" else None)


class TestSpacyDownloadCache:
    def test_install_uses_pip_when_available(self, monkeypatch, recorded_runs):
        _set_tools(monkeypatch, pip=True, uv="/usr/bin/uv")

        success, failed = download_cache.download_spacy_models(install=True)

        assert (success, failed) == (len(download_cache.SPACY_MODEL_WHEELS), [])
        assert all(
            cmd[:4] == [sys.executable, "-m", "pip", "install"] for cmd in recorded_runs
        )

    def test_install_falls_back_to_uv_without_pip(self, monkeypatch, recorded_runs):
        _set_tools(monkeypatch, pip=False, uv="/opt/uv/bin/uv")

        success, failed = download_cache.download_spacy_models(install=True)

        assert (success, failed) == (len(download_cache.SPACY_MODEL_WHEELS), [])
        assert [cmd[-1] for cmd in recorded_runs] == [
            url for _, url in sorted(download_cache.SPACY_MODEL_WHEELS.items())
        ]
        # --python pins the target to this interpreter's environment, not
        # whatever environment uv would discover from the working directory.
        assert all(
            cmd[:5] == ["/opt/uv/bin/uv", "pip", "install", "--python", sys.executable]
            for cmd in recorded_runs
        )

    def test_install_without_pip_or_uv_fails_without_running(
        self, monkeypatch, recorded_runs
    ):
        _set_tools(monkeypatch, pip=False, uv=None)

        success, failed = download_cache.download_spacy_models(install=True)

        assert success == 0
        assert recorded_runs == []
        assert sorted(name for name, _ in failed) == sorted(
            download_cache.SPACY_MODEL_WHEELS
        )
        assert "requirements-offline-smart-heading.txt" in failed[0][1]

    def test_download_without_pip_fails_even_with_uv(
        self, monkeypatch, recorded_runs, tmp_path
    ):
        # uv has no `pip download`, so the wheel download cannot use it.
        _set_tools(monkeypatch, pip=False, uv="/opt/uv/bin/uv")

        success, failed = download_cache.download_spacy_models(
            str(tmp_path), install=False
        )

        assert success == 0
        assert recorded_runs == []
        assert len(failed) == len(download_cache.SPACY_MODEL_WHEELS)
        assert "uv pip install pip" in failed[0][1]

    def test_download_uses_pip_download(self, monkeypatch, recorded_runs, tmp_path):
        _set_tools(monkeypatch, pip=True, uv=None)

        success, _ = download_cache.download_spacy_models(str(tmp_path), install=False)

        assert success == len(download_cache.SPACY_MODEL_WHEELS)
        assert all(
            cmd[:6] == [sys.executable, "-m", "pip", "download", "--no-deps", "--dest"]
            and cmd[6] == str(tmp_path)
            for cmd in recorded_runs
        )


# --------------------------------------------------------------------------- #
# tiktoken cache directory resolution
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(monkeypatch, tmp_path):
    for name in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    return fake_home


class _FakeEncoding:
    def encode(self, text):
        return [0]


@pytest.fixture
def seen_cache_dirs(monkeypatch):
    import tiktoken

    seen = []

    def fake_load(name):
        seen.append(os.environ.get("TIKTOKEN_CACHE_DIR"))
        return _FakeEncoding()

    monkeypatch.setattr(tiktoken, "encoding_for_model", fake_load)
    monkeypatch.setattr(tiktoken, "get_encoding", fake_load)
    return seen


class TestTiktokenDownloadCache:
    def test_default_is_home_cache_not_temp_dir(self, home, seen_cache_dirs, capsys):
        success, failed = download_cache.download_tiktoken_cache()

        expected = str(home / ".tiktoken_cache")
        assert (success, failed) == (len(seen_cache_dirs), [])
        assert (home / ".tiktoken_cache").is_dir()
        assert set(seen_cache_dirs) == {expected}
        assert "data-gym-cache" not in capsys.readouterr().out

    @pytest.mark.parametrize(
        "env_var, subdir, check_dir",
        [
            ("TIKTOKEN_CACHE_DIR", "from-env", True),
            ("DATA_GYM_CACHE_DIR", "data-gym", False),
        ],
    )
    def test_env_directory_is_honoured(
        self, home, monkeypatch, tmp_path, seen_cache_dirs, env_var, subdir, check_dir
    ):
        target = tmp_path / subdir
        monkeypatch.setenv(env_var, str(target))

        download_cache.download_tiktoken_cache(models=["cl100k_base"])

        assert seen_cache_dirs == [str(target)]
        if check_dir:
            assert target.is_dir()
        assert not (home / ".tiktoken_cache").exists()

    def test_explicit_directory_is_expanded_and_used(self, home, seen_cache_dirs):
        download_cache.download_tiktoken_cache("~/offline/tiktoken", ["cl100k_base"])

        expected = home / "offline" / "tiktoken"
        assert seen_cache_dirs == [str(expected)]
        assert expected.is_dir()

    def test_unresolvable_home_fails_instead_of_using_temp_dir(
        self, home, monkeypatch, seen_cache_dirs
    ):
        monkeypatch.setattr(os.path, "expanduser", lambda path: path)

        with pytest.raises(SystemExit) as excinfo:
            download_cache.download_tiktoken_cache()

        assert excinfo.value.code == 1
        assert seen_cache_dirs == []
