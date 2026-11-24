"""Tests for the UI customization bundle (workspace-entry PRD §8).

Covers: strict manifest schema (brand.logo required, explicit null accepted),
locale normalization (case folding, underscore rejection), single-level
fallbacks + default_locale, whole-bundle fail-fast validation, path
containment, MIME sniffing, the public read API (200 customized:false without
a bundle; no-store without ETag; hash-addressed immutable assets), revision /
hash change boundaries, orthogonality to workspace.html availability, and
/health remaining free of customization fields.
"""

import json
import logging
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.api.ui_customization import (
    UICustomizationError,
    UICustomizationSnapshot,
    UILocaleContent,
    WEBUI_CHROME_LOCALES,
    _sniff_logo_mime,
    chrome_language_for,
    load_ui_customization_snapshot,
    locale_direction,
    locales_without_chrome_translation,
    normalize_locale,
    resolve_ui_customization_snapshot,
)


pytestmark = pytest.mark.offline

PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_DEFAULT_UI",
    "ENABLE_API_DOCS",
    "UI_TEMPLATES_DIR",
)

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
)  # magic + padding; content check is magic-based
SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'


def _set_clean_env(mp):
    for var in _ENV_VARS_TO_ISOLATE:
        mp.delenv(var, raising=False)
    mp.setenv("AUTH_ACCOUNTS", "")
    mp.setenv("LIGHTRAG_API_KEY", "")
    mp.setenv("TOKEN_SECRET", "")
    mp.setenv("LLM_BINDING", "openai")
    mp.setenv("EMBEDDING_BINDING", "openai")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    _set_clean_env(monkeypatch)

    import lightrag.api.config as config

    saved = (config._global_args, config._initialized)
    config._global_args = None
    config._initialized = False
    yield
    config._global_args, config._initialized = saved


def make_bundle(root, manifest=None, *, extra_files=None):
    """Stage a valid two-locale bundle; `manifest` overrides the default."""
    bundle = root / "ui_templates"
    (bundle / "assets").mkdir(parents=True, exist_ok=True)
    (bundle / "locales" / "en").mkdir(parents=True, exist_ok=True)
    (bundle / "locales" / "zh").mkdir(parents=True, exist_ok=True)

    (bundle / "assets" / "logo.svg").write_bytes(SVG_BYTES)
    (bundle / "locales" / "en" / "welcome.md").write_text("# Welcome EN", "utf-8")
    (bundle / "locales" / "en" / "query_empty.md").write_text("Ask EN", "utf-8")
    (bundle / "locales" / "zh" / "welcome.md").write_text("# 欢迎", "utf-8")
    (bundle / "locales" / "zh" / "query_empty.md").write_text("请提问", "utf-8")

    if manifest is None:
        manifest = {
            "schema_version": 1,
            "default_locale": "en",
            # The fallback target must NOT be default_locale. Pointing "ko"
            # at "en" makes the manifest-fallback path and the default-locale
            # path land on the same answer.
            "fallbacks": {"ko": ["zh"]},
            "brand": {"logo": "assets/logo.svg"},
            "locales": {
                "en": {
                    "welcome": "locales/en/welcome.md",
                    "query_empty": "locales/en/query_empty.md",
                    "logo_alt": "ACME",
                },
                "zh": {
                    "welcome": "locales/zh/welcome.md",
                    "query_empty": "locales/zh/query_empty.md",
                    "logo_alt": "ACME 中文",
                },
            },
        }
    (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    for rel, content in (extra_files or {}).items():
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, "utf-8")
    return bundle


def add_login_consent(
    bundle,
    *,
    locale="zh",
    login="# 登录说明",
    agreements="# 协议",
    consent_documents=None,
):
    """Declare the login-page blurb and/or the merged agreement document."""
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    for field_name, text in (("login", login), ("agreements", agreements)):
        if text is None:
            manifest["locales"][locale].pop(field_name, None)
            continue
        rel = f"locales/{locale}/{field_name}.md"
        target = bundle / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, "utf-8")
        manifest["locales"][locale][field_name] = rel
    if consent_documents is not None:
        manifest["locales"][locale]["consent_documents"] = consent_documents
    (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return bundle


# --------------------------------------------------------------------------- #
# Snapshot loader
# --------------------------------------------------------------------------- #
class TestLocaleNormalization:
    def test_case_folding(self):
        assert normalize_locale("zh-tw") == "zh-TW"
        assert normalize_locale("ZH-hant-TW") == "zh-Hant-TW"
        assert normalize_locale("EN") == "en"

    def test_underscore_rejected(self):
        with pytest.raises(ValueError, match="underscore"):
            normalize_locale("zh_TW")

    @pytest.mark.parametrize("bad", ["", "a", "1x", "en-", "x" * 40])
    def test_invalid_forms_rejected(self, bad):
        with pytest.raises(ValueError):
            normalize_locale(bad)


class TestBundleValidation:
    def test_valid_bundle_loads(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        assert set(snapshot.locales) == {"en", "zh"}
        assert snapshot.default_locale == "en"
        assert snapshot.locales["en"].welcome == "# Welcome EN"
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        assert len(snapshot.assets) == 1
        logo = next(iter(snapshot.assets.values()))
        assert logo.mime == "image/svg+xml"
        assert snapshot.bundle_revision

    def test_explicit_null_logo_accepted(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["brand"]["logo"] = None
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].logo_asset_id is None
        assert snapshot.assets == {}

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda m: m.pop("brand"), "missing required"),
            (lambda m: m["brand"].pop("logo"), "missing required"),
            (lambda m: m.update(schema_version=2), "schema_version"),
            (lambda m: m.update(unknown_key=1), "unknown field"),
            (lambda m: m.update(default_locale="fr"), "not a declared locale"),
            (lambda m: m["locales"]["en"].pop("query_empty"), "missing required"),
            (lambda m: m["locales"]["en"].update(logo_alt=""), "logo_alt"),
            (lambda m: m["locales"]["en"].update(surprise=1), "unknown field"),
            (lambda m: m.update(fallbacks={"ko": []}), "non-empty array"),
            (lambda m: m.update(fallbacks={"ko": ["fr"]}), "not a declared locale"),
            (lambda m: m.update(fallbacks={"ko": ["en", "en"]}), "duplicate target"),
            (
                lambda m: m["locales"].update(
                    {
                        "zh_TW": {
                            "welcome": "locales/zh/welcome.md",
                            "query_empty": "locales/zh/query_empty.md",
                            "logo_alt": "x",
                        }
                    }
                ),
                "underscore",
            ),
            (
                lambda m: m["locales"].update(
                    {
                        "ZH-tw": {
                            "welcome": "locales/zh/welcome.md",
                            "query_empty": "locales/zh/query_empty.md",
                            "logo_alt": "x",
                        }
                    }
                ),
                "normalized form",
            ),
        ],
    )
    def test_schema_violations_fail(self, tmp_path, mutate, match):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        mutate(manifest)
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match=match):
            load_ui_customization_snapshot(bundle)

    def test_missing_manifest_fails(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").unlink()
        with pytest.raises(UICustomizationError, match="manifest.json is missing"):
            load_ui_customization_snapshot(bundle)

    def test_missing_referenced_file_fails(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "locales" / "zh" / "welcome.md").unlink()
        with pytest.raises(UICustomizationError, match="does not exist"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize(
        "path", ["/etc/hostname", "../outside.md", "locales/../../outside.md"]
    )
    def test_escaping_paths_rejected(self, tmp_path, path):
        (tmp_path / "outside.md").write_text("outside", "utf-8")
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["en"]["welcome"] = path
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError):
            load_ui_customization_snapshot(bundle)

    def test_symlink_escape_rejected(self, tmp_path):
        secret = tmp_path / "secret.md"
        secret.write_text("secret", "utf-8")
        bundle = make_bundle(tmp_path)
        link = bundle / "locales" / "en" / "link.md"
        link.symlink_to(secret)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["en"]["welcome"] = "locales/en/link.md"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="escapes"):
            load_ui_customization_snapshot(bundle)

    def test_template_size_limit(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "locales" / "en" / "welcome.md").write_text("x" * (64 * 1024 + 1))
        with pytest.raises(UICustomizationError, match="byte limit"):
            load_ui_customization_snapshot(bundle)

    def test_logo_mime_checked_by_content_through_bundle_loader(self, tmp_path):
        """End-to-end check that bundle loading wires invalid and valid SVGs through the sniffer."""
        bundle = make_bundle(tmp_path)
        # Valid SVG with XML prologue loads as image/svg+xml.
        (bundle / "assets" / "logo.svg").write_bytes(
            b'<?xml version="1.0"?>' + SVG_BYTES
        )
        snapshot = load_ui_customization_snapshot(bundle)
        assert next(iter(snapshot.assets.values())).mime == "image/svg+xml"

        # Invalid bytes with .svg extension fail startup.
        (bundle / "assets" / "logo.svg").write_bytes(b"GIF89a not allowed")
        with pytest.raises(UICustomizationError, match="not PNG, JPEG, WebP or SVG"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(
                b'<?xml version="1.0" encoding="UTF-8"?>\n'
                b"<config><value>42</value></config>",
                id="xml-declaration-without-svg-root",
            ),
            pytest.param(
                b"<!DOCTYPE html><html><body>not an image</body></html>",
                id="doctype-without-svg-root",
            ),
            pytest.param(b"<!-- just a comment -->", id="comment-only"),
            pytest.param(b"<svg/not-an-element>", id="slash-then-name"),
            pytest.param(b"<svg", id="bare-name-no-close"),
            pytest.param(b"<svg ", id="name-and-space-no-close"),
            pytest.param(
                b'<svg xmlns="http://www.w3.org/2000/svg"', id="attrs-no-close"
            ),
            pytest.param(b'<svg title=">', id="gt-inside-double-quotes"),
            pytest.param(b"<svg title='>'", id="gt-inside-single-quotes"),
            pytest.param(b'<svg title="' + b"x" * 5000, id="unterminated-quote-padded"),
            pytest.param(b"<svg foo\n<rect>", id="lt-inside-start-tag"),
            pytest.param(b'<svg t="a<b">', id="lt-inside-attribute-value"),
            pytest.param(b"<svg /x>", id="space-then-slash-then-name"),
            pytest.param(b"<svg width=100>", id="unquoted-attribute-value"),
            pytest.param(b'<svg a="1"b="2">', id="attributes-without-separator"),
            pytest.param(b"<svg foo>", id="attribute-without-value"),
            pytest.param(b'<svg ="1">', id="value-without-name"),
            pytest.param(b'<svg title="&"></svg>', id="bare-ampersand"),
            pytest.param(b'<svg title="&;"/>', id="empty-reference"),
            pytest.param(b'<svg title="&#xzz;"/>', id="malformed-hex-char-ref"),
            pytest.param(b'<svg title="&#12a;"/>', id="malformed-decimal-char-ref"),
            pytest.param(b'<svg title="&a b;"/>', id="whitespace-in-reference-name"),
            pytest.param(b"<svg\x0bfoo>", id="vertical-tab-is-not-xml-space"),
            pytest.param(b"<!-- brand mark", id="unterminated-comment"),
            pytest.param(b'<?xml version="1.0"', id="unterminated-pi"),
            pytest.param(
                b'<!DOCTYPE svg [<!ENTITY x "foo">' + SVG_BYTES,
                id="unterminated-doctype-subset",
            ),
        ],
    )
    def test_xml_without_svg_root_is_rejected(self, content):
        with pytest.raises(UICustomizationError, match="not PNG, JPEG, WebP or SVG"):
            _sniff_logo_mime(content, "assets/logo.svg")

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(SVG_BYTES, id="bare-svg-root"),
            pytest.param(
                b'<?xml version="1.0"?><!-- brand mark -->\n'
                b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "svg11.dtd">\n'
                + SVG_BYTES,
                id="declaration-comment-doctype-then-svg",
            ),
            pytest.param(b"\xef\xbb\xbf\n  " + SVG_BYTES, id="bom-and-whitespace"),
            pytest.param(b'<svg xmlns="http://www.w3.org/2000/svg"/>', id="empty-root"),
            pytest.param(
                b'<svg data-note="a>b" xmlns="http://www.w3.org/2000/svg">'
                b"<rect/></svg>",
                id="gt-in-attribute-value-then-real-close",
            ),
            pytest.param(
                b'<?xml-stylesheet type="text/css" href="a>b.css"?>' + SVG_BYTES,
                id="pi-pseudo-attribute-holds-gt",
            ),
            pytest.param(
                b'<!DOCTYPE svg SYSTEM "a>b.dtd">' + SVG_BYTES,
                id="doctype-system-literal-holds-gt",
            ),
            pytest.param(
                b'<!DOCTYPE svg [<!ENTITY x "foo">]>' + SVG_BYTES,
                id="doctype-internal-subset",
            ),
            pytest.param(
                b'<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" '
                b'"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [\n'
                b'<!ENTITY ns_extend "http://ns.adobe.com/Extensibility/1.0/">\n'
                b'<!ENTITY ns_ai "http://ns.adobe.com/AdobeIllustrator/10.0/">\n'
                b"]>" + SVG_BYTES,
                id="illustrator-multi-entity-subset",
            ),
            pytest.param(
                b"<!DOCTYPE svg [<!-- don't -->]>" + SVG_BYTES,
                id="subset-comment-with-apostrophe",
            ),
            pytest.param(b"<?ps don't?>" + SVG_BYTES, id="pi-with-lone-apostrophe"),
            pytest.param(b'<svg xmlns="x" />', id="space-before-empty-close"),
            pytest.param(b'<svg xmlns="x" ><rect/></svg>', id="space-before-close"),
            pytest.param(
                b'<svg\n  xmlns="http://www.w3.org/2000/svg"\n'
                b'  xmlns:xlink="http://www.w3.org/1999/xlink"\n'
                b'  version="1.1"\n  viewBox="0 0 24 24">\n<rect/></svg>',
                id="multiline-attribute-list",
            ),
            pytest.param(b'<svg xmlns = "x"><rect/></svg>', id="spaces-around-equals"),
            pytest.param(b"<svg xmlns='x'><rect/></svg>", id="single-quoted-value"),
            pytest.param(
                b'<svg style="fill:red;\n stroke:blue" xmlns="x"><rect/></svg>',
                id="newline-inside-attribute-value",
            ),
            pytest.param(
                b'<!DOCTYPE svg [<!ENTITY ns_x "http://x/">]>'
                b'<svg xmlns:x="&ns_x;" xmlns="x"/>',
                id="reference-to-subset-declared-entity",
            ),
            pytest.param(b'<svg t="&#169;" xmlns="x"/>', id="decimal-char-reference"),
            pytest.param(b'<svg t="&#x00A9;" xmlns="x"/>', id="hex-char-reference"),
            pytest.param(
                b'<svg t="&#X00a9;" xmlns="x"/>', id="hex-char-reference-upper"
            ),
            pytest.param(
                b"<!-- a [ \" ' ] brand mark -->" + SVG_BYTES,
                id="comment-with-brackets-and-quotes",
            ),
            pytest.param(
                b"<!--" + b"x" * 5000 + b"-->" + SVG_BYTES,
                id="long-prologue-does-not-hide-root",
            ),
        ],
    )
    def test_real_svg_variants_are_accepted(self, content):
        assert _sniff_logo_mime(content, "assets/logo.svg") == "image/svg+xml"

    def test_locale_logo_override(self, tmp_path):
        bundle = make_bundle(tmp_path, extra_files={"assets/logo-zh.png": PNG_BYTES})
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["zh"]["logo"] = "assets/logo-zh.png"
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].logo_asset_id == "logo-zh"
        assert snapshot.locales["en"].logo_asset_id == "brand-logo"
        mimes = {a.asset_id: a.mime for a in snapshot.assets.values()}
        assert mimes == {"brand-logo": "image/svg+xml", "logo-zh": "image/png"}


class TestCopyrightBundle:
    """`brand.copyright` with an optional per-locale override."""

    def test_absent_by_default(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        assert snapshot.locales["en"].copyright is None
        assert snapshot.locales["zh"].copyright is None

    def test_brand_level_applies_to_every_locale(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."
        assert snapshot.locales["zh"].copyright == "© 2025 ACME Inc."

    def test_locale_override_and_explicit_null(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        manifest["locales"]["zh"]["copyright"] = "© 2025 ACME 公司"
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].copyright == "© 2025 ACME 公司"
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."

        manifest["locales"]["zh"]["copyright"] = None
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].copyright is None
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."

    def test_blank_is_none_rather_than_a_startup_failure(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "   "
        manifest["locales"]["zh"]["copyright"] = "   "
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].copyright is None
        assert snapshot.locales["zh"].copyright is None

    def test_surrounding_whitespace_is_stripped(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "  © 2025 ACME Inc.  "
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["en"].copyright == "© 2025 ACME Inc."

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (
                lambda m: m["brand"].update(copyright=123),
                "brand.copyright must be a string or null",
            ),
            (
                lambda m: m["locales"]["zh"].update(copyright=["a"]),
                "locales.zh.copyright must be a string or null",
            ),
        ],
    )
    def test_non_string_is_rejected(self, tmp_path, mutate, match):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        mutate(manifest)
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        with pytest.raises(UICustomizationError, match=re.escape(match)):
            load_ui_customization_snapshot(bundle)

    def test_participates_in_the_bundle_revision(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle).bundle_revision
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        assert load_ui_customization_snapshot(bundle).bundle_revision != before


class TestLoginConsentBundle:
    """The login-page consent gate: BOTH optional templates or no gate."""

    def test_absent_by_default(self, tmp_path):
        snapshot = load_ui_customization_snapshot(make_bundle(tmp_path))
        content = snapshot.locales["zh"]
        assert content.login is None
        assert content.agreements is None
        assert content.consent_required is False

    def test_both_declared_turns_the_gate_on(self, tmp_path):
        bundle = add_login_consent(make_bundle(tmp_path))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].login == "# 登录说明"
        assert snapshot.locales["zh"].agreements == "# 协议"
        assert snapshot.locales["zh"].consent_required is True
        assert snapshot.locales["en"].consent_required is False

    @pytest.mark.parametrize(
        "login, agreements",
        [
            ("# 登录说明", None),
            (None, "# 协议"),
        ],
    )
    def test_half_a_configuration_leaves_the_gate_off(
        self, tmp_path, login, agreements
    ):
        bundle = add_login_consent(
            make_bundle(tmp_path), login=login, agreements=agreements
        )
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_required is False

    def test_explicit_null_is_undeclared(self, tmp_path):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"].update(login=None, agreements=None)
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_required is False

    @pytest.mark.parametrize("field_name", ["login", "agreements"])
    def test_blank_template_is_rejected(self, tmp_path, field_name):
        bundle = add_login_consent(make_bundle(tmp_path), **{field_name: "   \n"})
        with pytest.raises(UICustomizationError, match=f"locales.zh.{field_name}"):
            load_ui_customization_snapshot(bundle)

    @pytest.mark.parametrize("field_name", ["login", "agreements"])
    def test_optional_templates_obey_the_required_ones_rules(
        self, tmp_path, field_name
    ):
        for bad_path, match in (
            ("../../etc/passwd", "traversal"),
            ("/etc/passwd", "absolute"),
            ("locales/zh/missing.md", "does not exist"),
        ):
            bundle = make_bundle(tmp_path / bad_path.replace("/", "_"))
            manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
            manifest["locales"]["zh"][field_name] = bad_path
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            with pytest.raises(UICustomizationError, match=match):
                load_ui_customization_snapshot(bundle)

        oversized = make_bundle(tmp_path / "oversized")
        (oversized / "locales" / "zh" / "big.md").write_text("x" * 70000, "utf-8")
        manifest = json.loads((oversized / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"][field_name] = "locales/zh/big.md"
        (oversized / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="byte limit"):
            load_ui_customization_snapshot(oversized)

    @pytest.mark.parametrize("field_name", ["login", "agreements"])
    def test_non_string_path_is_rejected(self, tmp_path, field_name):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"][field_name] = 42
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(UICustomizationError, match="path string or null"):
            load_ui_customization_snapshot(bundle)


class TestConsentDocumentsName:
    def test_declared_value_is_carried(self, tmp_path):
        bundle = add_login_consent(
            make_bundle(tmp_path), consent_documents="《示例公司服务条款》"
        )
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_documents == "《示例公司服务条款》"
        assert snapshot.locales["en"].consent_documents is None

    def test_explicit_null_is_undeclared(self, tmp_path):
        bundle = add_login_consent(make_bundle(tmp_path), consent_documents=None)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"]["consent_documents"] = None
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_documents is None
        assert snapshot.locales["zh"].consent_required is True

    def test_it_does_not_switch_the_gate_on(self, tmp_path):
        bundle = add_login_consent(
            make_bundle(tmp_path),
            login=None,
            agreements=None,
            consent_documents="《服务条款》",
        )
        snapshot = load_ui_customization_snapshot(bundle)
        assert snapshot.locales["zh"].consent_documents == "《服务条款》"
        assert snapshot.locales["zh"].consent_required is False

    @pytest.mark.parametrize("bad", ["   \n", 42])
    def test_blank_or_non_string_is_rejected(self, tmp_path, bad):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["locales"]["zh"]["consent_documents"] = bad
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        with pytest.raises(
            UICustomizationError, match="consent_documents must be a non-empty string"
        ):
            load_ui_customization_snapshot(bundle)

    def test_consent_templates_participate_in_the_revision(self, tmp_path):
        bundle = add_login_consent(make_bundle(tmp_path))
        before = load_ui_customization_snapshot(bundle)
        (bundle / "locales" / "zh" / "agreements.md").write_text("# 协议 v2", "utf-8")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision


class TestRevisionBoundaries:
    def test_deterministic_across_loads(self, tmp_path):
        bundle = make_bundle(tmp_path)
        first = load_ui_customization_snapshot(bundle)
        second = load_ui_customization_snapshot(bundle)
        assert first.bundle_revision == second.bundle_revision
        assert {k: v.sha256 for k, v in first.assets.items()} == {
            k: v.sha256 for k, v in second.assets.items()
        }

    def test_text_change_moves_revision_but_not_logo_hash(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle)
        (bundle / "locales" / "zh" / "welcome.md").write_text("# 新欢迎", "utf-8")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision
        assert set(before.assets) == set(after.assets)

    def test_logo_change_moves_both(self, tmp_path):
        bundle = make_bundle(tmp_path)
        before = load_ui_customization_snapshot(bundle)
        (bundle / "assets" / "logo.svg").write_bytes(SVG_BYTES + b"<!--v2-->")
        after = load_ui_customization_snapshot(bundle)
        assert before.bundle_revision != after.bundle_revision
        assert set(before.assets) != set(after.assets)
        assert {a.asset_id for a in after.assets.values()} == {"brand-logo"}


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class _FakeLightRAG:
    """Minimal LightRAG double satisfying create_app's route wiring."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


class _FakeOllamaAPI:
    """Minimal OllamaAPI double exposing an empty router."""

    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _stage_frontend(tmp_path, *, workspace=True):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(exist_ok=True)
    (webui_dir / "index.html").write_text(
        f"<html><head>{PLACEHOLDER}</head><body>index</body></html>", "utf-8"
    )
    if workspace:
        (webui_dir / "workspace.html").write_text(
            f"<html><head>{PLACEHOLDER}</head><body>workspace</body></html>", "utf-8"
        )


def _build_app(tmp_path, monkeypatch, *cli_args):
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", *cli_args]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )
    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        lightrag_server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )
    return lightrag_server.create_app(args)


class TestCustomizationEndpointNoBundle:
    @pytest.fixture(scope="class")
    @classmethod
    def no_bundle_client(cls, tmp_path_factory):
        import lightrag.api.config as config

        root = tmp_path_factory.mktemp("no_bundle_ui")
        _stage_frontend(root)
        saved = (config._global_args, config._initialized)
        with pytest.MonkeyPatch.context() as mp:
            _set_clean_env(mp)
            mp.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
            app = _build_app(root, mp)
            try:
                yield TestClient(app)
            finally:
                config._global_args, config._initialized = saved

    def test_customized_false_with_brand(self, no_bundle_client):
        resp = no_bundle_client.get("/ui/customization?locale=zh-TW")
        assert resp.status_code == 200
        data = resp.json()
        assert data["customized"] is False
        assert data["brand"]["title"]
        assert "welcome" not in data
        assert "copyright" not in data["brand"]
        assert not data.get("consent_required")
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_asset_endpoint_rejects_everything(self, no_bundle_client):
        assert (
            no_bundle_client.get(
                "/ui/customization/assets/abc123/brand-logo"
            ).status_code
            == 404
        )


class TestCustomizationEndpointWithBundle:
    @pytest.fixture(scope="class")
    @classmethod
    def bundle_client(cls, tmp_path_factory):
        """Read-only app booted once for the standard two-locale bundle."""
        import lightrag.api.config as config

        root = tmp_path_factory.mktemp("with_bundle_ui")
        bundle = make_bundle(root)
        _stage_frontend(root, workspace=True)
        saved = (config._global_args, config._initialized)
        with pytest.MonkeyPatch.context() as mp:
            _set_clean_env(mp)
            mp.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
            mp.setenv("UI_TEMPLATES_DIR", str(bundle))
            app = _build_app(root, mp)
            try:
                yield TestClient(app)
            finally:
                config._global_args, config._initialized = saved

    def _client(self, tmp_path, monkeypatch, *cli_args, workspace=True, consent=False):
        bundle = make_bundle(tmp_path)
        if consent:
            add_login_consent(bundle)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace)
        return TestClient(_build_app(tmp_path, monkeypatch, *cli_args))

    def test_full_representation(self, bundle_client):
        resp = bundle_client.get("/ui/customization?locale=zh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["customized"] is True
        assert data["locale"] == "zh"
        assert data["fallback_used"] is False
        assert data["direction"] == "ltr"
        assert data["welcome"] == {"format": "markdown", "content": "# 欢迎"}
        assert data["query_empty"]["content"] == "请提问"
        assert data["brand"]["logo_alt"] == "ACME 中文"
        assert data["brand"]["logo_url"].startswith("/ui/customization/assets/")
        assert data["brand"]["copyright"] is None
        assert data["login"] is None
        assert data["agreements"] is None
        assert data["consent_documents"] is None
        assert data["consent_required"] is False
        assert resp.headers["cache-control"] == "no-store"
        assert "etag" not in resp.headers

    def test_login_consent_served_and_follows_resolved_locale(
        self, tmp_path, monkeypatch
    ):
        client = self._client(tmp_path, monkeypatch, consent=True)
        zh = client.get("/ui/customization?locale=zh").json()
        assert zh["login"] == {"format": "markdown", "content": "# 登录说明"}
        assert zh["agreements"] == {"format": "markdown", "content": "# 协议"}
        assert zh["consent_required"] is True
        assert zh["consent_documents"] is None

        # "en" is declared but carries neither template.
        en = client.get("/ui/customization?locale=en").json()
        assert en["locale"] == "en"
        assert en["consent_required"] is False
        assert en["agreements"] is None

        # "ko" falls back to "zh", which carries both.
        ko = client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("zh", True)
        assert ko["consent_required"] is True
        assert ko["agreements"]["content"] == "# 协议"

    def test_consent_documents_served(self, tmp_path, monkeypatch):
        bundle = add_login_consent(
            make_bundle(tmp_path), consent_documents="《示例公司服务条款》"
        )
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=True)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        data = client.get("/ui/customization?locale=zh").json()
        assert data["consent_documents"] == "《示例公司服务条款》"
        assert (
            client.get("/ui/customization?locale=en").json()["consent_documents"]
            is None
        )

    def test_copyright_served_per_resolved_locale(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
        manifest["brand"]["copyright"] = "© 2025 ACME Inc."
        manifest["locales"]["zh"]["copyright"] = "© 2025 ACME 公司"
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))

        assert (
            client.get("/ui/customization?locale=en").json()["brand"]["copyright"]
            == "© 2025 ACME Inc."
        )
        ko = client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("zh", True)
        assert ko["brand"]["copyright"] == "© 2025 ACME 公司"

    def test_case_normalization_and_underscore_rejection(self, bundle_client):
        ok = bundle_client.get("/ui/customization?locale=ZH")
        assert ok.json()["locale"] == "zh"
        bad = bundle_client.get("/ui/customization?locale=zh_TW")
        assert bad.status_code == 400

    def test_fallback_chain_then_default(self, bundle_client):
        ko = bundle_client.get("/ui/customization?locale=ko").json()
        assert (ko["locale"], ko["fallback_used"]) == ("zh", True)

        fr = bundle_client.get("/ui/customization?locale=fr").json()
        assert (fr["locale"], fr["fallback_used"]) == ("en", True)
        assert fr["customized"] is True
        assert ko["locale"] != fr["locale"]

    @pytest.mark.parametrize(
        ("locale", "direction"),
        [
            ("ar", "rtl"),
            ("he", "rtl"),
            ("fa", "rtl"),
            ("ur", "rtl"),
            ("ps", "rtl"),
            ("ckb", "rtl"),
            ("dv", "rtl"),
            ("yi", "rtl"),
            ("nqo", "rtl"),
            ("syr", "rtl"),
            ("ku-Arab", "rtl"),
            ("ku", "ltr"),
            ("az-Arab", "rtl"),
            ("az", "ltr"),
            ("pa-Arab", "rtl"),
            ("pa", "ltr"),
            ("ar-Latn", "ltr"),
            ("ar-u-nu-latn", "rtl"),
            ("ar-aao-Latn", "ltr"),
            ("ar-aao-arz-acm-Latn", "ltr"),
            ("ar-aao-arz-acm-apc-Latn", "rtl"),
            ("ar-EG", "rtl"),
            ("en", "ltr"),
            ("zh-TW", "ltr"),
            ("fr", "ltr"),
            ("ru", "ltr"),
            ("und", "ltr"),
        ],
    )
    def test_locale_direction_registry(self, locale, direction):
        assert locale_direction(locale) == direction

    def test_rtl_direction_from_registry(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["ar"] = {
            "welcome": "locales/en/welcome.md",
            "query_empty": "locales/en/query_empty.md",
            "logo_alt": "ACME AR",
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization?locale=ar").json()["direction"] == "rtl"

    def test_asset_served_immutable_with_security_headers(self, bundle_client):
        logo_url = bundle_client.get("/ui/customization?locale=en").json()["brand"][
            "logo_url"
        ]
        resp = bundle_client.get(logo_url)
        assert resp.status_code == 200
        assert resp.content == SVG_BYTES
        assert resp.headers["content-type"].startswith("image/svg+xml")
        assert "immutable" in resp.headers["cache-control"]
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" in resp.headers
        parts = logo_url.rsplit("/", 2)
        assert bundle_client.get(f"{parts[0]}/{'0' * 64}/{parts[2]}").status_code == 404
        assert bundle_client.get(f"{parts[0]}/{parts[1]}/other-id").status_code == 404

    def test_logo_url_keeps_api_prefix(self, tmp_path, monkeypatch):
        client = self._client(tmp_path, monkeypatch, "--api-prefix", "/site01")
        data = client.get("/site01/ui/customization?locale=en").json()
        assert data["brand"]["logo_url"].startswith("/site01/ui/customization/assets/")
        assert client.get(data["brand"]["logo_url"]).status_code == 200

    def test_health_carries_no_customization_fields(self, bundle_client):
        health = bundle_client.get("/health")
        assert "bundle_revision" not in health.text
        assert "customization" not in health.text

    def test_route_audit_includes_mounts_and_customization(self, bundle_client):
        app = bundle_client.app
        mount_paths = {getattr(route, "path", None) for route in app.routes}
        assert "/webui" in mount_paths
        assert "/workspace" in mount_paths
        api_paths = set(app.openapi()["paths"])
        assert "/ui/customization" in api_paths
        assert "/ui/customization/assets/{asset_hash}/{asset_id}" in api_paths


class TestOrthogonality:
    """workspace.html presence × UI_TEMPLATES_DIR unset/valid/invalid:
    customization behavior depends ONLY on the latter dimension."""

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_invalid_bundle_fails_startup_regardless_of_products(
        self, tmp_path, monkeypatch, workspace_present
    ):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").write_text("{broken", "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace_present)
        with pytest.raises(UICustomizationError):
            _build_app(tmp_path, monkeypatch)

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_valid_bundle_loads_regardless_of_products(
        self, tmp_path, monkeypatch, workspace_present
    ):
        bundle = make_bundle(tmp_path)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))
        _stage_frontend(tmp_path, workspace=workspace_present)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization?locale=en").json()["customized"] is True

    @pytest.mark.parametrize("workspace_present", [True, False])
    def test_unset_is_normal_in_both_product_states(
        self, tmp_path, monkeypatch, workspace_present
    ):
        _stage_frontend(tmp_path, workspace=workspace_present)
        client = TestClient(_build_app(tmp_path, monkeypatch))
        assert client.get("/ui/customization").json()["customized"] is False


class TestChromeTranslationCoverage:
    def test_the_python_list_matches_the_webui_language_list(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "lightrag_webui"
            / "src"
            / "lib"
            / "browserLanguage.ts"
        ).read_text(encoding="utf-8")
        block = re.search(r"SUPPORTED_UI_LANGUAGES\s*=\s*\[(.*?)\]", source, re.S)
        assert block is not None, "SUPPORTED_UI_LANGUAGES not found in the WebUI source"
        webui_ids = set(re.findall(r"'([^']+)'", block.group(1)))
        assert webui_ids, "no languages parsed from the WebUI source"
        assert {locale.replace("_", "-") for locale in webui_ids} == set(
            WEBUI_CHROME_LOCALES
        )

    @pytest.mark.parametrize(
        "locale,expected",
        [
            ("de-AT", "de"),
            ("zh-CN", "zh"),
            ("zh-HK", "zh-TW"),
            ("zh-Hant", "zh-TW"),
            ("es", None),
            ("", None),
        ],
    )
    def test_chrome_language_for_mirrors_the_frontend_matcher(self, locale, expected):
        assert chrome_language_for(locale) == expected

    def test_only_the_untranslated_locales_are_named(self):
        assert locales_without_chrome_translation(
            ["en", "es", "zh-TW", "he", "de"]
        ) == ["es", "he"]
        assert locales_without_chrome_translation(["en", "fr"]) == []

    def test_startup_warns_about_a_bundle_the_chrome_cannot_follow(
        self, tmp_path, monkeypatch, lightrag_log_records
    ):
        _stage_frontend(tmp_path)
        bundle = make_bundle(tmp_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        manifest["locales"]["es"] = manifest["locales"].pop("zh")
        manifest["default_locale"] = "es"
        manifest["fallbacks"] = {}
        (bundle / "manifest.json").write_text(json.dumps(manifest), "utf-8")
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))

        _build_app(tmp_path, monkeypatch)

        warnings = [
            record.getMessage()
            for record in lightrag_log_records
            if record.levelno >= logging.WARNING
        ]
        assert any("no interface translation" in message for message in warnings)
        assert any("'es'" in message for message in warnings)

    def test_a_fully_translatable_bundle_warns_about_nothing(
        self, tmp_path, monkeypatch, lightrag_log_records
    ):
        _stage_frontend(tmp_path)
        bundle = make_bundle(tmp_path)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(bundle))

        _build_app(tmp_path, monkeypatch)

        assert not any(
            "no interface translation" in record.getMessage()
            for record in lightrag_log_records
        )


class TestLogoUrlAltCoupling:
    def _app(self, snapshot):
        from fastapi import FastAPI
        from lightrag.api.routers.ui_customization_routes import (
            create_ui_customization_routes,
        )

        app = FastAPI()
        app.include_router(create_ui_customization_routes(snapshot, "ACME", "desc"))
        return app

    @pytest.mark.parametrize(
        "logo_asset_id",
        [
            pytest.param("brand-logo", id="missing-referenced-asset"),
            pytest.param(None, id="explicit-no-logo"),
        ],
    )
    def test_alt_is_dropped_when_no_logo_url(self, logo_asset_id):
        snapshot = UICustomizationSnapshot(
            default_locale="en",
            locales={
                "en": UILocaleContent(
                    welcome="w",
                    query_empty="q",
                    logo_alt="ACME",
                    logo_asset_id=logo_asset_id,
                )
            },
            fallbacks={},
            assets={},
            bundle_revision="deadbeef",
        )
        data = TestClient(self._app(snapshot)).get("/ui/customization").json()
        assert data["brand"]["logo_url"] is None
        assert data["brand"]["logo_alt"] is None


# --------------------------------------------------------------------------- #
# Unpopulated bundle directory
# --------------------------------------------------------------------------- #
class TestUnpopulatedBundleDirectory:
    @pytest.mark.parametrize(
        "populate",
        [
            pytest.param(lambda root: None, id="empty-dir"),
            pytest.param(
                lambda root: (root / ".DS_Store").write_bytes(b"whatever"),
                id="stray-ds-store",
            ),
            pytest.param(
                lambda root: (
                    (root / "locales" / "en").mkdir(parents=True),
                    (root / "locales" / "en" / "welcome.md").write_text("# hi", "utf-8"),
                ),
                id="partial-without-manifest",
            ),
        ],
    )
    def test_directory_without_manifest_resolves_to_no_bundle(self, tmp_path, populate):
        root = tmp_path / "ui_templates"
        root.mkdir()
        populate(root)
        assert resolve_ui_customization_snapshot(root) is None

    @pytest.mark.parametrize(
        ("prepare", "match"),
        [
            pytest.param(
                lambda base: base / "nowhere",
                "does not exist",
                id="missing-directory",
            ),
            pytest.param(
                lambda base: (
                    (base / "ui_templates").write_text("", "utf-8"),
                    base / "ui_templates",
                )[1],
                "is not a directory",
                id="file-instead-of-directory",
            ),
            pytest.param(
                lambda base: (
                    (base / "ui_templates" / "manifest.json").mkdir(parents=True),
                    base / "ui_templates",
                )[1],
                "not a readable regular",
                id="manifest-is-directory",
            ),
            pytest.param(
                lambda base: (
                    (base / "ui_templates").mkdir(),
                    (base / "ui_templates" / "manifest.json").symlink_to(
                        base / "ui_templates" / "nowhere.json"
                    ),
                    base / "ui_templates",
                )[2],
                "not a readable regular",
                id="dangling-manifest-symlink",
            ),
        ],
    )
    def test_invalid_bundle_paths_fail(self, tmp_path, prepare, match):
        target = prepare(tmp_path)
        with pytest.raises(UICustomizationError, match=match):
            resolve_ui_customization_snapshot(target)

    def test_a_manifest_symlink_that_resolves_is_loaded(self, tmp_path):
        bundle = make_bundle(tmp_path)
        real = bundle / "manifest.real.json"
        (bundle / "manifest.json").rename(real)
        (bundle / "manifest.json").symlink_to(real)
        assert resolve_ui_customization_snapshot(bundle) is not None

    def test_a_present_manifest_keeps_every_validation(self, tmp_path):
        bundle = make_bundle(tmp_path)
        (bundle / "manifest.json").write_text("{broken", "utf-8")
        with pytest.raises(UICustomizationError, match="not valid JSON"):
            resolve_ui_customization_snapshot(bundle)

        bundle2 = make_bundle(tmp_path / "second")
        (bundle2 / "locales" / "zh" / "welcome.md").unlink()
        with pytest.raises(UICustomizationError, match="does not exist"):
            resolve_ui_customization_snapshot(bundle2)

    def test_loader_contract_is_unchanged(self, tmp_path):
        empty = tmp_path / "ui_templates"
        empty.mkdir()
        with pytest.raises(UICustomizationError, match="manifest.json is missing"):
            load_ui_customization_snapshot(empty)

    def test_startup_survives_an_unpopulated_mount_and_says_so(
        self, tmp_path, monkeypatch, lightrag_log_records
    ):
        _stage_frontend(tmp_path)
        mount_point = tmp_path / "data" / "ui_templates"
        mount_point.mkdir(parents=True)
        monkeypatch.setenv("UI_TEMPLATES_DIR", str(mount_point))

        app = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        assert client.get("/ui/customization").json()["customized"] is False

        warnings = [
            record.getMessage()
            for record in lightrag_log_records
            if record.levelno >= logging.WARNING
        ]
        assert any("no manifest.json" in message for message in warnings)
        assert any(str(mount_point) in message for message in warnings)
