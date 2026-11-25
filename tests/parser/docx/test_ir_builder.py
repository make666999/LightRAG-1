"""Tests for ``NativeDocxIRBuilder``: ``doc_title`` resolution and drawing security."""

from __future__ import annotations

import pytest

from lightrag.parser.docx.ir_builder import NativeDocxIRBuilder

pytestmark = pytest.mark.offline


def _normalize(
    content: str = "正文一句。", *, heading: str = "前言", parse_metadata=None
):
    return NativeDocxIRBuilder().normalize(
        [
            {
                "uuid": "p1",
                "uuid_end": "p1",
                "heading": heading,
                "content": content,
                "parent_headings": [],
                "level": 1,
            }
        ],
        document_name="doc.docx",
        asset_dir_name="doc.blocks.assets",
        parse_metadata=parse_metadata,
    )


# ---------------------------------------------------------------------------
# doc_title resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parse_metadata", "expected"),
    [
        pytest.param(
            {"doc_title": "某某管理办法", "first_heading": "前言"},
            "某某管理办法",
            id="explicit-doc-title-wins",
        ),
        pytest.param(
            {"doc_title": "", "first_heading": "前言"},
            "",
            id="explicit-empty-beats-fallback",
        ),
        pytest.param(
            {"first_heading": "前言"},
            "前言",
            id="legacy-first-heading-chain",
        ),
        pytest.param({}, "doc", id="legacy-stem-empty-dict"),
        pytest.param(None, "doc", id="legacy-stem-none"),
    ],
)
def test_doc_title_resolution(parse_metadata, expected: str) -> None:
    """Explicit ``parse_metadata['doc_title']`` wins outright (even when empty);
    without the key the legacy chain holds: ``first_heading``, else file stem."""
    assert _normalize(parse_metadata=parse_metadata).doc_title == expected


# ---------------------------------------------------------------------------
# Drawing asset security & external-src handling
# ---------------------------------------------------------------------------


def test_native_docx_ir_accepts_safe_local_drawing_asset() -> None:
    ir = _normalize(
        '<drawing id="1" name="fig" format="png" path="doc.blocks.assets/fig.png" />',
        heading="Section",
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == "fig.png"
    assert len(ir.assets) == 1
    assert ir.assets[0].ref == "fig.png"
    assert ir.assets[0].suggested_name == "fig.png"
    assert ir.assets[0].source is None


@pytest.mark.parametrize(
    "path",
    [
        "doc.blocks.assets/../secret.png",
        "doc.blocks.assets//tmp/secret.png",
        r"doc.blocks.assets/..\secret.png",
    ],
)
def test_native_docx_ir_rejects_unsafe_local_drawing_asset(path: str) -> None:
    ir = _normalize(
        f'<drawing id="1" name="fig" format="png" path="{path}" />',
        heading="Section",
    )

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert ir.assets == []


@pytest.mark.parametrize(
    ("tag", "expected_src", "expected_fmt"),
    [
        pytest.param(
            '<drawing id="1" name="fig" format="gif" src="../images/legacy.gif" />',
            "../images/legacy.gif",
            "gif",
            id="explicit-format-passthrough",
        ),
        pytest.param(
            '<drawing id="1" name="fig" src="../images/legacy.gif" />',
            "../images/legacy.gif",
            "gif",
            id="inferred-format-from-src",
        ),
        pytest.param(
            # _infer_format_from_target returns None -> coerced to "", never "None".
            '<drawing id="1" name="fig" src="https://example.com/image" />',
            "https://example.com/image",
            "",
            id="extensionless-src-empty-format",
        ),
        pytest.param(
            # ``path`` is local-only: a remote URL in ``path`` is dropped, not salvaged.
            '<drawing id="1" name="fig" format="png" path="https://example.com/x.png" />',
            "",
            "png",
            id="non-asset-path-dropped-without-salvage",
        ),
    ],
)
def test_native_docx_ir_external_and_non_asset_drawings(
    tag: str, expected_src: str, expected_fmt: str
) -> None:
    ir = _normalize(tag, heading="Section")

    drawing = ir.blocks[0].drawings[0]
    assert drawing.asset_ref == ""
    assert drawing.src == expected_src
    assert drawing.fmt == expected_fmt
    assert ir.assets == []
