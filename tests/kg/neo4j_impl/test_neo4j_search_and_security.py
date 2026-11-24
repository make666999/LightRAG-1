"""Offline search, full-text index collision, and Cypher injection tests for Neo4j.

Covers:
- lossy full-text index name collision refusal across workspaces;
- CJK / Hangul / kana Lucene query formatting and case-insensitive fallback;
- Lucene reserved-character sanitization in ``_sanitize_fulltext_query``;
- workspace label backtick escaping, raw parameter binding, and APOC isolation.
"""

from __future__ import annotations

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage, WorkspaceIndexCollisionError
from tests.kg._cypher_fakes import FakeDriver, FakeResult

pytestmark = pytest.mark.offline


def _make_storage(
    workspace: str = "test",
    *,
    driver: FakeDriver | None = None,
    database: str | None = "neo4j",
    namespace: str = "test",
    global_config: dict | None = None,
) -> Neo4JStorage:
    storage = Neo4JStorage(
        namespace=namespace,
        global_config={} if global_config is None else global_config,
        embedding_func=None,
        workspace=workspace if workspace else None,
    )
    if driver is not None:
        storage._driver = driver
        storage._DATABASE = database
    return storage


# ---------------------------------------------------------------------------
# Full-text index name collision across workspaces
# ---------------------------------------------------------------------------


def _sent_ddl(queries: list[str]) -> tuple[bool, bool]:
    """Return ``(sent DROP INDEX, sent CREATE FULLTEXT INDEX)``."""
    return (
        any(query.strip().startswith("DROP INDEX") for query in queries),
        any("CREATE FULLTEXT INDEX" in query for query in queries),
    )


def _index_driver(show_rows: list[dict]) -> FakeDriver:
    def _route(query: str, _params: dict) -> FakeResult:
        if query.strip().startswith("SHOW FULLTEXT INDEXES"):
            return FakeResult(show_rows)
        return FakeResult([])

    return FakeDriver(handler=_route)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (None, (False, True)),  # no index yet: create it
        ("ONLINE", (False, False)),  # own index online: leave it alone
        ("POPULATING", (True, True)),  # own index not online: recreate it
    ],
)
async def test_own_or_missing_index_is_handled_as_before(state, expected):
    storage = _make_storage("foo-bar")
    label = storage._get_workspace_label()
    rows = []
    if state is not None:
        rows.append(
            {
                "name": storage._get_fulltext_index_name(label),
                "state": state,
                "labelsOrTypes": [storage._get_raw_workspace_label()],
            }
        )
    driver = _index_driver(rows)

    await storage._create_fulltext_index(driver, "neo4j", label)

    assert _sent_ddl(driver.queries) == expected


@pytest.mark.parametrize("state", ["ONLINE", "POPULATING"])
async def test_index_of_another_workspace_is_refused_untouched(state):
    owner = _make_storage("foo-bar")
    claimant = _make_storage("foo_bar")
    claimant_label = claimant._get_workspace_label()
    index_name = claimant._get_fulltext_index_name(claimant_label)
    assert owner._get_fulltext_index_name(owner._get_workspace_label()) == index_name
    driver = _index_driver(
        [
            {
                "name": index_name,
                "state": state,
                "labelsOrTypes": [owner._get_raw_workspace_label()],
            }
        ]
    )

    # Raised from inside the function's own catch-all ``except Exception``,
    # pinning that the collision refusal is never swallowed there.
    with pytest.raises(WorkspaceIndexCollisionError, match="foo-bar"):
        await claimant._create_fulltext_index(driver, "neo4j", claimant_label)

    # A non-ONLINE index would otherwise be dropped and recreated.
    assert _sent_ddl(driver.queries) == (False, False)


# ---------------------------------------------------------------------------
# search_labels CJK analyzer routing and fallback
# ---------------------------------------------------------------------------


def _search_storage(*, fail_fulltext: bool = False) -> tuple[Neo4JStorage, FakeDriver]:
    def _route(query: str, _params: dict) -> FakeResult:
        if fail_fulltext and "db.index.fulltext.queryNodes" in query:
            raise RuntimeError("full-text index unavailable")
        return FakeResult([])

    driver = FakeDriver(handler=_route)
    storage = _make_storage(
        "test",
        driver=driver,
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
    )
    return storage, driver


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("北京大学", True),
        ("삼성전자", True),  # Hangul syllables
        ("LG전자", True),  # mixed Latin + Hangul
        ("ㅅㅅ", True),  # Hangul compatibility jamo
        ("タワー", True),  # Katakana only
        ("ひらがな", True),  # Hiragana only
        ("ﾀﾜｰ", True),  # Halfwidth Katakana
        ("𠀀", True),  # CJK Extension B (supplementary plane)
        ("Samsung", False),
        ("ＡＢＣ", False),  # Fullwidth Latin is not bigrammed by the analyzer
        ("", False),
    ],
)
def test_is_cjk_text(text, expected):
    storage, _ = _search_storage()
    assert storage._is_cjk_text(text) is expected


@pytest.mark.parametrize("query", ["삼성전자", "서"])
async def test_search_labels_sends_cjk_query_as_terms_plus_prefix(query):
    storage, driver = _search_storage()

    await storage.search_labels(query)

    assert len(driver.calls) == 1
    assert driver.calls[0][1]["search_query"] == f"{query} {query}*"


async def test_search_labels_keeps_prefix_wildcard_for_latin_query():
    storage, driver = _search_storage()

    await storage.search_labels("Samsung")

    assert len(driver.calls) == 1
    assert driver.calls[0][1]["search_query"] == "Samsung*"


async def test_fallback_matches_case_insensitively_for_every_script():
    """A mixed query routed to the CJK path keeps the lowercased CONTAINS."""
    query = "lg전자"
    storage, driver = _search_storage(fail_fulltext=True)

    await storage.search_labels(query)

    assert len(driver.calls) == 2  # full-text attempt, then CONTAINS fallback
    fallback_query, fallback_params = driver.calls[1]
    assert "toLower(n.entity_id) AS label_lower" in fallback_query
    assert "label_lower CONTAINS $query_lower" in fallback_query
    assert fallback_params == {"query_lower": query.lower(), "limit": 50}


# ---------------------------------------------------------------------------
# Lucene reserved-character sanitization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Reserved characters are replaced with spaces and whitespace collapsed.
        ("tb-", "tb"),
        ("node:x", "node x"),
        ("a (b)", "a b"),
        ("C++", "C"),
        ("foo && bar", "foo bar"),
        ("name*", "name"),
        # Composed entirely of reserved characters -> empty (caller returns []).
        ("---", ""),
        ("()[]{}", ""),
        # No reserved characters -> unchanged (modulo whitespace collapsing).
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_sanitize_fulltext_query(raw, expected):
    assert Neo4JStorage._sanitize_fulltext_query(raw) == expected


# ---------------------------------------------------------------------------
# Workspace label escaping and APOC injection guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workspace", "expected"),
    [
        # Every backtick is doubled, not only the first.
        ("a`b`c", "a``b``c"),
        # A breakout attempt stays inside the quoted identifier.
        ("ws`} RETURN 0", "ws``} RETURN 0"),
        # Single quotes are literal inside backticks and must not be doubled.
        ("ws'`name", "ws'``name"),
    ],
)
def test_get_workspace_label(workspace, expected):
    storage = _make_storage(workspace)
    assert storage._get_workspace_label() == expected


@pytest.mark.parametrize(
    ("workspace", "expected"),
    [
        ("  trimmed  ", "trimmed"),
        ("", "base"),
        ("   ", "base"),
        # No escaping of any kind: the raw value is bound, never interpolated.
        ("ws`name", "ws`name"),
        ("x'); CALL apoc.util.sleep(1);", "x'); CALL apoc.util.sleep(1);"),
    ],
)
def test_get_raw_workspace_label(workspace, expected):
    storage = _make_storage(workspace)
    assert storage._get_raw_workspace_label() == expected


@pytest.mark.asyncio
async def test_subgraph_query_has_no_label_filter():
    """The APOC subgraph call must never carry or interpolate the workspace label."""
    payload = "x'); CALL apoc.util.sleep(1);"
    driver = FakeDriver(
        records=[{"total_nodes": 0, "node_info": [], "relationships": []}]
    )
    storage = _make_storage(payload, driver=driver, database=None)

    await storage.get_knowledge_graph(node_label="some-entity", max_depth=2)

    subgraph_calls = [(q, p) for q, p in driver.calls if "apoc.path.subgraphAll" in q]
    assert subgraph_calls, "expected an apoc.path.subgraphAll query"

    for query, params in subgraph_calls:
        assert "labelFilter:" not in query
        assert "label_filter" not in params
        apoc_config = query[query.index("apoc.path.subgraphAll") :]
        assert payload not in apoc_config
