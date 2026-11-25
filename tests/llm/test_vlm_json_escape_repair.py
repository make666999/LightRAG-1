"""Regression tests: LaTeX escape damage repair for VLM JSON responses.

Models writing LaTeX inside JSON strings routinely under-escape backslashes:
``"\\frac"`` is valid JSON meaning form feed + ``rac``, so ``json_repair.loads``
silently decodes it and destroys the command. ``repair_vlm_json_escape_damage``
restores the zero-risk control characters (form feed / backspace + letter)
everywhere, and restores whitespace-class residues (``\\t``, ``\\r``, ``\\n``)
only inside explicit dollar math outside Markdown code regions.
"""

import logging

import json_repair
import pytest

from lightrag.utils import (
    _WS_LATEX_MATH_PATTERN,
    _WS_LATEX_SUSPECT_PATTERN,
    repair_vlm_json_escape_damage,
)

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Table 1: Repairs applied (and idempotent)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("damaged", "expected"),
    [
        # Zero-risk control characters (form feed / backspace + letter).
        pytest.param("$\x0crac{610}{C}$", r"$\frac{610}{C}$", id="formfeed-frac"),
        pytest.param(
            "$\x08eta + \x08ar{x}$", r"$\beta + \bar{x}$", id="backspace-beta-bar"
        ),
        # Whitespace-class commands inside inline / display math.
        pytest.param("domain is $\tau^2$", r"domain is $\tau^2$", id="inline-tau"),
        pytest.param("$$\nabla f = 0$$", r"$$\nabla f = 0$$", id="display-nabla"),
        pytest.param("$\rho + \right)$", r"$\rho + \right)$", id="inline-rho-right"),
        pytest.param(
            "first $\tau^2$, second $$a \times b$$",
            r"first $\tau^2$, second $$a \times b$$",
            id="multiple-spans",
        ),
        # Word-character continuations (_, digits) must not block in-math repair.
        pytest.param("$\rho_{ij}$", r"$\rho_{ij}$", id="math-subscript-rho"),
        pytest.param("$\theta_0$", r"$\theta_0$", id="math-subscript-theta"),
        pytest.param("$\times2$", r"$\times2$", id="math-digit-times"),
        pytest.param("$$\nabla_x f$$", r"$$\nabla_x f$$", id="display-subscript-nabla"),
        # Line-break commands survive the inline line-break veto.
        pytest.param(
            "$\nabla f$", "$" + r"\nabla" + " f$", id="inline-nabla-veto-exempt"
        ),
        pytest.param(
            "$x \notin A$", "$x " + r"\notin" + " A$", id="inline-notin-veto-exempt"
        ),
        pytest.param(
            "$\rho + 1$", "$" + r"\rho" + " + 1$", id="inline-rho-veto-exempt"
        ),
        pytest.param(
            "$x \right)$", "$x " + r"\right" + ")$", id="inline-right-veto-exempt"
        ),
        # Display math spans across LF, CR, and CRLF line breaks.
        pytest.param(
            "$$\n\tau^2\n$$", "$$\n" + r"\tau^2" + "\n$$", id="display-lf-lines"
        ),
        pytest.param("$$\n\nabla f$$", "$$\n" + r"\nabla f$$", id="display-lf-nabla"),
        pytest.param("$$a +\n\tau$$", "$$a +\n" + r"\tau$$", id="display-lf"),
        pytest.param("$$a +\r\tau$$", "$$a +\r" + r"\tau$$", id="display-cr"),
        pytest.param("$$a +\r\n\tau$$", "$$a +\r\n" + r"\tau$$", id="display-crlf"),
        # Currency / stray dollar recovery.
        pytest.param(
            "Cost is $5. The domain is $\tau^2$ end.",
            "Cost is $5. The domain is " + r"$\tau^2$" + " end.",
            id="currency-before-inline-math",
        ),
        pytest.param(
            "$5-$10 range $\tau$",
            "$5-$10 range " + r"$\tau$",
            id="currency-range-before-math",
        ),
        pytest.param(
            "$10 \times 5$ then $\tau$",
            r"$10 \times 5$ then $\tau$",
            id="intact-math-before-damaged-span",
        ),
        pytest.param(
            "a $\tau$$ b",
            "a " + r"$\tau$" + "$ b",
            id="inline-closes-against-first-of-double-dollar",
        ),
        pytest.param(
            "Cost $5. Formula $$\tau^2$$",
            "Cost $5. Formula " + r"$$\tau^2$$",
            id="currency-before-display-math",
        ),
        pytest.param(
            "Cost $5 and\text is $\tau$",
            "Cost $5 and\text is " + r"$\tau$",
            id="currency-prose-stays-untouched-before-math",
        ),
        pytest.param(
            "$$ broken and $\tau$ here",
            "$$ broken and " + r"$\tau$" + " here",
            id="unclosed-display-skipped-before-inline",
        ),
        # Math outside or after Markdown code regions is still repaired.
        pytest.param(
            "stray ` tick then $\tau$ ok",
            "stray ` tick then " + r"$\tau$" + " ok",
            id="unclosed-single-backtick-does-not-block",
        ),
        pytest.param(
            '```sh\n```not a closer\necho "$HOME"\n\text=1\necho "$PATH"\n```\nafter $\tau$',
            '```sh\n```not a closer\necho "$HOME"\n\text=1\necho "$PATH"\n```\nafter '
            + r"$\tau$",
            id="after-fence-with-fake-inner-closer",
        ),
        pytest.param(
            "```foo`bar\nafter $\tau$",
            "```foo`bar\nafter " + r"$\tau$",
            id="backtick-in-backtick-fence-info-string",
        ),
        pytest.param(
            "```sh\na\n```\nafter $\tau$",
            "```sh\na\n```\nafter " + r"$\tau$",
            id="after-fence-lf",
        ),
        pytest.param(
            "```sh\r\na\r\n```\r\nafter $\tau$",
            "```sh\r\na\r\n```\r\nafter " + r"$\tau$",
            id="after-fence-crlf",
        ),
        pytest.param(
            '````sh\necho "$A"\n\text=1\necho "$B"\n`````\nthen $\tau$',
            '````sh\necho "$A"\n\text=1\necho "$B"\n`````\nthen ' + r"$\tau$",
            id="closing-fence-longer-than-opener",
        ),
        pytest.param(
            '```sh\necho "$HOME"\n```\nthen $\tau$ here',
            '```sh\necho "$HOME"\n```\nthen ' + r"$\tau$" + " here",
            id="after-closed-code-block",
        ),
        pytest.param(
            "A stray ` tick.\n\nvalue $\tau$ here\n\nanother ` tick.",
            "A stray ` tick.\n\nvalue $" + r"\tau" + "$ here\n\nanother ` tick.",
            id="paragraphs-between-stray-backticks",
        ),
        pytest.param(
            "Use \\`foo, then $\tau$ and \\`bar",
            "Use \\`foo, then $" + r"\tau" + "$ and \\`bar",
            id="escaped-backticks-do-not-shield-math",
        ),
        pytest.param(
            "x `` unmatched $\tau$ then ` end",
            "x `` unmatched " + r"$\tau$" + " then ` end",
            id="unmatched-double-backtick-does-not-open-span",
        ),
        # Known boundary: single-line unquoted shell without quotes/backticks.
        pytest.param(
            "A=$X; \text=1; B=$Y",
            "A=$X; " + r"\text" + "=1; B=$Y",
            id="unquoted-single-line-shell-boundary",
        ),
    ],
)
def test_repairs_and_idempotency(damaged, expected):
    once = repair_vlm_json_escape_damage(damaged)
    assert once == expected
    assert repair_vlm_json_escape_damage(once) == expected


# --------------------------------------------------------------------------- #
# Table 2: Inputs that must stay completely unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("before\x0c after\x08.", id="isolated-control-chars"),
        pytest.param(
            r"$\frac{a}{b}$ and \beta with plain text", id="already-clean-latex"
        ),
        pytest.param("$5\rangle and $10", id="between-two-currency-amounts"),
        pytest.param(
            "$ x $ then\text column $y$", id="prose-between-padded-and-real-span"
        ),
        pytest.param("成本为$a$，领域\tau为$b$", id="cjk-prose-between-math-spans"),
        pytest.param("Unclosed $$x and\text$ suffix", id="unclosed-display-delimiter"),
        pytest.param("price $5 then\tau", id="unpaired-dollar"),
        pytest.param(r"escaped \$" + "\tau" + "$", id="escaped-dollar"),
        # Inline content gate: quotes and non-residue line breaks (LF/CR/CRLF).
        pytest.param('$\text{"x"} + \tau$', id="double-quote-inside-math"),
        pytest.param("A=$X\n\text=1\nB=$Y", id="inline-veto-code-lf"),
        pytest.param("A=$X\r\text=1\rB=$Y", id="inline-veto-code-cr"),
        pytest.param("A=$X\r\n\text=1\r\nB=$Y", id="inline-veto-code-crlf"),
        pytest.param("see $a +\n\tau$ end", id="inline-veto-prose-lf"),
        pytest.param("see $a +\r\tau$ end", id="inline-veto-prose-cr"),
        pytest.param("see $a +\r\n\tau$ end", id="inline-veto-prose-crlf"),
        # Markdown code blocks & inline spans protected verbatim.
        pytest.param(
            '```sh\necho "$HOME"\n\text=1\necho "$PATH"\n```', id="fenced-block-lf"
        ),
        pytest.param(
            '```sh\r\necho "$HOME"\r\n\text=1\r\necho "$PATH"\r\n```',
            id="fenced-block-crlf",
        ),
        pytest.param(
            'inline `echo "$HOME"; \text=1; echo "$PATH"` done', id="inline-code-span"
        ),
        pytest.param("text\n```sh\n$A \text $B\n", id="unclosed-fence-protects-rest"),
        pytest.param("```foo\nafter $\tau$", id="unclosed-backtick-fence-info"),
        pytest.param(
            "~~~foo`bar\nafter $\tau$", id="unclosed-tilde-fence-with-backtick-info"
        ),
        pytest.param(
            '```sh\n```not a closer\necho "$HOME"\n\text=1\necho "$PATH"\n```',
            id="fake-closer-with-info-string",
        ),
        pytest.param(
            "```sh\n\t```\nA=$X; \text=1; B=$Y\n```", id="tab-indented-fake-closer"
        ),
        pytest.param(
            "   ```sh\n   A=$X; \text=1; B=$Y\n   ```", id="three-space-indented-fence"
        ),
        pytest.param(
            'see `A ``B "$HOME" \text=1 "$PATH"` end',
            id="inline-closer-bounded-both-sides",
        ),
        pytest.param('echo "$HOME"; \text=1; echo "$PATH"', id="unmarked-quoted-shell"),
        pytest.param(
            'stray ` tick\n```sh\nrun `cmd` now\necho "$HOME"\n\text=1\necho "$PATH"\n```',
            id="stray-tick-before-fence-with-backticks",
        ),
        pytest.param(
            'Example:\n\n    echo "$HOME"\n    \text=1\n    echo "$PATH"\n\ndone',
            id="four-space-indented-code",
        ),
        pytest.param(
            "A stray ` tick.\n\n```sh\necho `date` ; A=$X; \text=1; B=$Y\n```\n",
            id="bare-stray-tick-cannot-steal-fence",
        ),
        pytest.param(
            "A stray \\` tick.\n\n```sh\necho `date` ; A=$X; \text=1; B=$Y\n```\n",
            id="escaped-stray-tick-cannot-steal-fence",
        ),
        pytest.param(
            "A stray ` tick.\n\nRun `A=$X; \text=1; B=$Y` now.",
            id="stray-tick-cannot-cross-lf-blank-line",
        ),
        pytest.param(
            "A stray ` tick.\r\n\r\nRun `A=$X; \text=1; B=$Y` now.",
            id="stray-tick-cannot-cross-crlf-blank-line",
        ),
        pytest.param(
            "A stray ` tick.\n   \nRun `A=$X; \text=1; B=$Y` now.",
            id="stray-tick-cannot-cross-spaces-blank-line",
        ),
        pytest.param(
            "Run `A=$X;\n\text=1; B=$Y` now.",
            id="inline-code-span-crosses-single-newline",
        ),
        pytest.param(
            "Type \\`x then `A=$X; \text=1; B=$Y`",
            id="escaped-backtick-does-not-open-span",
        ),
        # Parity check: an even run of backslashes immediately before ` leaves a real opener.
        pytest.param(
            "Path C:\\\\`A=$X; \text=1; B=$Y`",
            id="doubled-backslash-adjacent-leaves-real-opener",
        ),
        pytest.param(
            "`A=$X; \text=1; B=$Y\\`",
            id="escaped-backtick-still-closes-inline-span",
        ),
    ],
)
def test_protected_and_vetoed_inputs_stay_unchanged(text):
    assert repair_vlm_json_escape_damage(text) == text


# --------------------------------------------------------------------------- #
# Table 3: Logging on in-math repairs vs outside-math suspects vs clean prose
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("damaged", "expected", "expected_snippet"),
    [
        pytest.param("domain is $\tau^2$", r"domain is $\tau^2$", r"\\tau^2", id="tau"),
        pytest.param("$$\nabla f = 0$$", r"$$\nabla f = 0$$", None, id="nabla"),
        pytest.param("$\rho + \right)$", r"$\rho + \right)$", None, id="rho-right"),
        pytest.param("$\rho_{ij}$", r"$\rho_{ij}$", None, id="subscript-rho"),
        pytest.param("$\theta_0$", r"$\theta_0$", None, id="subscript-theta"),
        pytest.param("$\times2$", r"$\times2$", None, id="digit-times"),
        pytest.param("$$\nabla_x f$$", r"$$\nabla_x f$$", None, id="display-nabla-sub"),
    ],
)
def test_in_math_repair_logs_warning_and_no_unrepaired_warning(
    damaged, expected, expected_snippet, lightrag_caplog
):
    with lightrag_caplog.at_level(logging.WARNING):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")

    assert result == expected
    assert any(
        "Repaired whitespace-class LaTeX escape damage inside dollar math"
        in rec.message
        and "table/t1" in rec.getMessage()
        for rec in lightrag_caplog.records
    )
    if expected_snippet is not None:
        assert any(
            expected_snippet in rec.getMessage() for rec in lightrag_caplog.records
        )
    assert not any(
        "not auto-repaired" in rec.message for rec in lightrag_caplog.records
    )


@pytest.mark.parametrize(
    "damaged",
    [
        pytest.param("label:\tau", id="end-of-string"),
        pytest.param("阈值\tau为0.5", id="cjk-follower-tau"),
        pytest.param("算子\nabla作用于 f", id="cjk-follower-nabla"),
        pytest.param("密度\rho的分布", id="cjk-follower-rho"),
        pytest.param("阈值\tau。", id="ideographic-stop"),
        pytest.param("参数\tilde，如下", id="fullwidth-comma"),
        pytest.param("the \tau value", id="space-follower"),
        pytest.param("$5 costs \rho.", id="ascii-punctuation-follower"),
        pytest.param("$ \tau $", id="padded-inline-span-residue"),
        pytest.param("价格$5，公式$\tau$为", id="cjk-stray-dollar-residue"),
    ],
)
def test_outside_math_damage_is_logged_not_rewritten(damaged, lightrag_caplog):
    with lightrag_caplog.at_level(logging.WARNING):
        result = repair_vlm_json_escape_damage(damaged, context="table/t1")

    assert result == damaged
    assert any(
        "not auto-repaired" in rec.message and "table/t1" in rec.getMessage()
        for rec in lightrag_caplog.records
    )


@pytest.mark.parametrize(
    "legit",
    [
        pytest.param("col1\tauthor list", id="english-word-after-tab"),
        pytest.param("col\text_id header", id="underscore-after-residue"),
        pytest.param("the \tau2 value", id="digit-after-residue"),
        pytest.param("row\rightmost column", id="word-after-cr-residue"),
        pytest.param(
            "col1\tauthor list\nablation studies follow\nexists in the table",
            id="multiline-ordinary-prose",
        ),
    ],
)
def test_legitimate_whitespace_is_silent(legit, lightrag_caplog):
    with lightrag_caplog.at_level(logging.WARNING):
        result = repair_vlm_json_escape_damage(legit)

    assert result == legit
    assert not lightrag_caplog.records


def test_in_math_pattern_stays_a_superset_of_the_prose_pattern():
    probes = [
        f"{residue}{tail}"
        for residue, tails in (
            ("\t", ("au", "heta", "imes", "ext", "ilde", "herefore", "riangle")),
            ("\r", ("ho", "ight", "angle", "ceil")),
            ("\n", ("abla", "otin")),
        )
        for tail in tails
    ]
    followers = ["", " ", ".", "_", "2", "x", "为", "。", "Δ"]
    mismatches = [
        f"prefix{probe}{follower}"
        for probe in probes
        for follower in followers
        if _WS_LATEX_SUSPECT_PATTERN.search(f"prefix{probe}{follower}")
        and not _WS_LATEX_MATH_PATTERN.search(f"prefix{probe}{follower}")
    ]
    assert mismatches == []


# --------------------------------------------------------------------------- #
# Integration tests: json_repair, prompt contracts, and extraction wiring
# --------------------------------------------------------------------------- #


def test_mixed_escaping_real_world_response():
    raw_response = (
        '{"name": "成本对比", '
        '"description": "GraphRAG消耗$\\frac{610 \\\\times 1,000}{C_{\\\\text{max}}}$次调用"}'
    )
    parsed = json_repair.loads(raw_response)
    assert isinstance(parsed, dict)
    description = parsed["description"]
    assert isinstance(description, str)
    assert "\x0c" in description

    repaired = repair_vlm_json_escape_damage(description)
    assert "$\\frac{610 \\times 1,000}{C_{\\text{max}}}$" in repaired
    assert "\x0c" not in repaired


def test_prompts_require_double_escaped_backslashes():
    from lightrag.prompt import PROMPTS
    from lightrag.prompt_multimodal import MULTIMODAL_PROMPTS

    for key in ("image_analysis", "table_analysis", "equation_analysis"):
        template = MULTIMODAL_PROMPTS[key]
        assert "escape backslashes" in template or "double-escaped" in template, (
            f"{key}: missing backslash escaping rule in OUTPUT RULES"
        )

    extraction_template = PROMPTS["entity_extraction_json_system_prompt"]
    assert (
        "escape backslashes" in extraction_template
        and "double-escaped" in extraction_template
    )


@pytest.mark.asyncio
async def test_extraction_json_result_repairs_latex_escape_damage():
    from lightrag.operate import _process_json_extraction_result

    raw_response = (
        '{"entities": [{"name": "LightRAG", "type": "Other", '
        '"description": "成本为 $\\frac{610}{C}$，领域为 $\\tau^2$"}], '
        '"relationships": [{"source": "LightRAG", "target": "GraphRAG", '
        '"keywords": "cost", '
        '"description": "比较 $\\frac{a}{b}$、$a \\times b$ 与 $\\\\beta$"}]}'
    )

    nodes, edges = await _process_json_extraction_result(
        raw_response, chunk_key="chunk-test", timestamp=0
    )

    (entity_list,) = [nodes[k] for k in nodes if k == "LightRAG"]
    assert "\\frac{610}{C}" in entity_list[0]["description"]
    assert "\\tau^2" in entity_list[0]["description"]
    assert "\x0c" not in entity_list[0]["description"]
    assert "\t" not in entity_list[0]["description"]

    (edge_list,) = list(edges.values())
    assert "\\frac{a}{b}" in edge_list[0]["description"]
    assert "\\times" in edge_list[0]["description"]
    assert "\\beta" in edge_list[0]["description"]
    assert "\x0c" not in edge_list[0]["description"]
