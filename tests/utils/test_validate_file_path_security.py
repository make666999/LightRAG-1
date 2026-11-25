"""Tests for ``validate_file_path_security`` path-traversal containment.

The helper is the containment boundary for callers turning externally-influenced
strings into filesystem paths (upload/scan routes and sidecar artifact
resolution). Assertions compare ``Path`` objects and use ``relative_to`` rather
than string matching because separators and resolution differ across OSes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.utils import validate_file_path_security

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("raw_path", "expected_relative"),
    [
        pytest.param("report.pdf", Path("report.pdf"), id="plain-filename"),
        pytest.param(
            "sub/dir/report.pdf", Path("sub/dir/report.pdf"), id="nested-relative"
        ),
        pytest.param("./report.pdf", Path("report.pdf"), id="leading-dot-segment"),
        pytest.param("a/./b.pdf", Path("a/b.pdf"), id="interior-dot-segment"),
        pytest.param(
            r"sub\dir\report.pdf",
            Path("sub/dir/report.pdf"),
            id="backslashes-normalized",
        ),
        pytest.param("  report.pdf  ", Path("report.pdf"), id="surrounding-whitespace"),
        # Containment is decided on the resolved path, so contained ".." stays valid.
        pytest.param("a/../b.pdf", Path("b.pdf"), id="contained-traversal"),
        # Existence is intentionally not checked: absent-but-contained returns a Path.
        pytest.param(
            "nope/missing.pdf", Path("nope/missing.pdf"), id="absent-but-contained"
        ),
        # "." resolves to the base directory itself, which is contained.
        pytest.param(".", Path("."), id="exact-base-directory"),
    ],
)
def test_accepted_paths_resolve_inside_base(
    tmp_path: Path, raw_path: str, expected_relative: Path
) -> None:
    result = validate_file_path_security(raw_path, tmp_path)

    assert result is not None
    assert result.relative_to(tmp_path.resolve()) == expected_relative
    if raw_path == "nope/missing.pdf":
        assert not result.exists()


@pytest.mark.parametrize(
    "unsafe_path",
    [
        pytest.param("../etc/passwd", id="posix-single-parent"),
        pytest.param("../../etc/passwd", id="posix-double-parent"),
        pytest.param("a/../../escape.pdf", id="posix-nested-escape"),
        pytest.param("sub/../../escape.pdf", id="posix-sub-escape"),
        pytest.param(r"..\windows\system32", id="windows-single-parent"),
        pytest.param(r"..\..\secret.txt", id="windows-double-parent"),
        pytest.param(r"sub\..\..\escape.pdf", id="windows-sub-escape"),
        pytest.param("..", id="bare-parent"),
        pytest.param("/etc/passwd", id="absolute-path"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="spaces"),
        pytest.param("\t", id="tab"),
        pytest.param("\n", id="newline"),
        pytest.param(None, id="none"),
    ],
)
def test_rejected_paths_return_none(tmp_path: Path, unsafe_path: str | None) -> None:
    assert validate_file_path_security(unsafe_path, tmp_path) is None


def test_sibling_directory_sharing_name_prefix_is_rejected(tmp_path: Path) -> None:
    """``/base_evil`` must not pass containment against base ``/base``."""
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "base_evil").mkdir()

    assert validate_file_path_security("../base_evil/loot.txt", base) is None
