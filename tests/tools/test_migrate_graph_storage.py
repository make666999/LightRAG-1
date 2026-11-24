"""Offline unit tests for the pure graph-migration helpers.

Covers the Phase 1 (AGE -> PGTable) migration invariants:
canonical ordering parity with PGTableGraphStorage (Python min/max, incl.
non-ASCII), fail-closed detection of reciprocal directed edges and duplicate
node ids, the parallel-edge backstop, derived written-node sets for
compensation, and source-driven verification.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from lightrag.tools.migrate_graph_storage import (
    MigrationDataError,
    MigrationPreconditionError,
    MigrationWriteError,
    VerificationReport,
    _check_resolved_workspace,
    _precheck_requested_workspace,
    canonicalize_edge,
    compensate_partial_migration,
    count_parallel_edges,
    derive_written_node_ids,
    detect_duplicate_node_ids,
    detect_self_loop_edges,
    detect_reciprocal_pairs,
    find_non_jsonb_representable,
    dry_run_migration,
    edge_properties,
    main,
    migrate_graph,
    payloads_equal,
    sort_directed_edges,
    target_slice_is_empty,
    verify_source_side,
)

pytestmark = pytest.mark.offline


_UNSET = object()  # distinguishes "return None" from "not configured"


def _edge(source, target, **props):
    """AGE-shaped enumerated edge: properties merged with source/target."""
    return {**props, "source": source, "target": target}


# ---------------------------------------------------------------------------
# canonicalize_edge — must match PGTable's Python min/max exactly
# ---------------------------------------------------------------------------


class TestCanonicalizeEdge:
    def test_non_ascii_code_point_order(self):
        # The collation trap: PostgreSQL LEAST/GREATEST under e.g. en_US.utf8
        # would order some of these pairs differently than Python code points.
        # The tool must reproduce Python min/max bit-for-bit.
        assert canonicalize_edge("한국", "일본") == (
            min("한국", "일본"),
            max("한국", "일본"),
        )
        assert canonicalize_edge("Über", "Uber") == ("Uber", "Über")
        assert canonicalize_edge("é", "z") == ("z", "é")  # é (U+00E9) > z (U+007A)


# ---------------------------------------------------------------------------
# payloads_equal — shared type-strict comparison rule
# ---------------------------------------------------------------------------


class TestPayloadsEqual:
    def test_type_strict_scalars(self):
        # dict == would pass all three of these; the shared rule must not.
        assert not payloads_equal(True, 1)
        assert not payloads_equal(1, 1.0)
        assert not payloads_equal("1", 1)
        assert payloads_equal(1, 1)
        assert payloads_equal(True, True)
        assert payloads_equal(None, None)

    def test_nested_structures(self):
        a = {"meta": {"weight": 1.0, "tags": ["x", "y"]}, "n": 2}
        b = {"n": 2, "meta": {"tags": ["x", "y"], "weight": 1.0}}
        assert payloads_equal(a, b)
        assert not payloads_equal(
            a, {"meta": {"weight": 1, "tags": ["x", "y"]}, "n": 2}
        )
        assert not payloads_equal({"tags": ["x"]}, {"tags": ["x", "y"]})

    def test_nan_never_equal(self):
        # Safe-direction false positive, documented in the module.
        assert not payloads_equal(float("nan"), float("nan"))

    def test_tuple_wrapper_cannot_defeat_type_gate(self):
        # Regression: tuples previously fell through to leaf ==, so a tuple
        # wrapper defeated the whole recursive type gate, nested dicts
        # included. Tuples must recurse like lists.
        assert not payloads_equal(({"x": True},), ({"x": 1},))
        assert not payloads_equal((True,), (1,))
        assert payloads_equal(({"x": 1},), ({"x": 1},))

    def test_unknown_types_fail_closed(self):
        # Non-JSON types never reach leaf ==: their __eq__ may coerce
        # ({True} == {1} is True), so they compare unequal unconditionally.
        assert not payloads_equal({True}, {1})
        assert not payloads_equal({1}, {1})


# ---------------------------------------------------------------------------
# count_parallel_edges — backstop for the AGE ordered-pair invariant
# ---------------------------------------------------------------------------


class TestCountParallelEdges:
    def test_backstop_fires_on_parallel_rows(self):
        # Two rows for the SAME ordered pair with different payloads (identical
        # rows would already have collapsed under the AGE query's DISTINCT).
        edges = [
            _edge("a", "b", w=1),
            _edge("a", "b", w=2),
            _edge("x", "y", w=1),
        ]
        counts = count_parallel_edges(edges)
        assert counts[("a", "b")] == 2
        assert counts[("x", "y")] == 1
        violations = {k: c for k, c in counts.items() if c > 1}
        assert violations == {("a", "b"): 2}

    def test_reciprocal_pair_is_not_a_violation(self):
        # a->b plus b->a share a canonical key but are DISTINCT ordered pairs;
        # they belong to detect_reciprocal_pairs, not this backstop.
        edges = [_edge("a", "b", w=1), _edge("b", "a", w=2)]
        counts = count_parallel_edges(edges)
        assert counts == {("a", "b"): 1, ("b", "a"): 1}
        assert not any(c > 1 for c in counts.values())


# ---------------------------------------------------------------------------
# derive_written_node_ids — compensation set under the empty-target invariant
# ---------------------------------------------------------------------------


class TestDeriveWrittenNodeIds:
    def test_includes_auto_created_endpoints(self):
        # upsert_edges_batch silently auto-creates missing endpoints; ids never
        # present in the migrated node list must still be in the written set.
        migrated = ["a", "b"]
        edges = [_edge("a", "b"), _edge("b", "ghost"), _edge("phantom", "phantom")]
        written = derive_written_node_ids(migrated, edges)
        assert written == {"a", "b", "ghost", "phantom"}


# ---------------------------------------------------------------------------
# sort_directed_edges — determinism (AC-c)
# ---------------------------------------------------------------------------


class TestSortDirectedEdges:
    def test_deterministic_across_enumeration_orders(self):
        edges = [
            _edge("b", "c", w=2),
            _edge("a", "b", w=1),
            _edge("a", "b", w=0),  # parallel row: tiebreaker keeps order total
            _edge("한국", "일본", w=3),
        ]
        import random

        shuffled = list(edges)
        random.Random(42).shuffle(shuffled)
        assert sort_directed_edges(edges) == sort_directed_edges(shuffled)

    def test_sorted_by_source_then_target(self):
        edges = [_edge("b", "a"), _edge("a", "z"), _edge("a", "b")]
        ordered = sort_directed_edges(edges)
        assert [(e["source"], e["target"]) for e in ordered] == [
            ("a", "b"),
            ("a", "z"),
            ("b", "a"),
        ]


# ---------------------------------------------------------------------------
# verify_source_side — source-driven comparison
# ---------------------------------------------------------------------------


class TestVerifySourceSide:
    def _clean_migration(self):
        # Source: AGE-shaped, one directed row per pair. A reciprocal — even
        # with identical payloads — is refused now, so a "clean" source has
        # none: collapsing one would change degree on the way into the target.
        source_nodes: list[dict[str, Any]] = [
            {"id": "a", "entity_id": "a", "entity_type": "person"},
            {"id": "b", "entity_id": "b", "entity_type": "place"},
            {"id": "c", "entity_id": "c", "entity_type": "thing"},
        ]
        source_edges = [
            _edge("a", "b", weight=1.0),
            _edge("c", "a", weight=2.0),
        ]
        # Target: canonicalized, each undirected edge once.
        target_nodes = [dict(n) for n in source_nodes]
        target_edges = [
            _edge("a", "b", weight=1.0),
            _edge("a", "c", weight=2.0),
        ]
        return source_nodes, source_edges, target_nodes, target_edges

    def test_clean_migration_ok(self):
        report = verify_source_side(*self._clean_migration())
        assert isinstance(report, VerificationReport)
        assert report.ok

    def test_missing_node_and_edge(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        report = verify_source_side(
            source_nodes, source_edges, target_nodes[:-1], target_edges[:-1]
        )
        assert not report.ok
        assert report.missing_node_ids == ["c"]
        assert report.missing_edges == [("a", "c")]

    def test_unexpected_target_rows(self):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        target_nodes = target_nodes + [{"id": "stray", "entity_id": "stray"}]
        target_edges = target_edges + [_edge("a", "stray", w=9)]
        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        assert report.unexpected_node_ids == ["stray"]
        assert report.unexpected_edges == [("a", "stray")]

    @pytest.mark.parametrize(
        "kind, expected",
        [
            pytest.param(
                "edge_value",
                [("a", "b", {"weight": 1.0}, {"weight": 99.0})],
                id="edge_value",
            ),
            pytest.param(
                "node_value",
                [
                    (
                        "a",
                        {"entity_id": "a", "entity_type": "person"},
                        {"entity_id": "a", "entity_type": "MUTATED"},
                    )
                ],
                id="node_value",
            ),
            pytest.param("node_type_strict", ["a"], id="node_type_strict"),
            pytest.param("edge_type_strict", [("a", "c")], id="edge_type_strict"),
        ],
    )
    def test_property_mismatch(self, kind, expected):
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        if kind == "edge_value":
            target_edges[0] = _edge("a", "b", weight=99.0)
        elif kind == "node_value":
            target_nodes[0] = {"id": "a", "entity_id": "a", "entity_type": "MUTATED"}
        elif kind == "node_type_strict":
            source_nodes[0]["flag"] = True
            target_nodes[0]["flag"] = 1
        elif kind == "edge_type_strict":
            target_edges[1] = _edge("a", "c", weight=2)  # source has 2.0

        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        if kind == "edge_value":
            key, src_props, tgt_props = report.edge_property_mismatches[0]
            assert len(report.edge_property_mismatches) == 1
            assert (key[0], key[1], src_props, tgt_props) == expected[0]
        elif kind == "node_value":
            assert report.node_property_mismatches == expected
        elif kind == "node_type_strict":
            assert [m[0] for m in report.node_property_mismatches] == expected
        else:
            assert [m[0] for m in report.edge_property_mismatches] == expected

    def test_parallel_source_rows_fail_verification_in_both_orders(self):
        # Regression: parallel source rows [a->b w=1, a->b w=2] against a
        # target holding only w=2 previously yielded ok=True in one
        # enumeration order and ok=False in the other. Verify must run the
        # backstop itself and fail closed in BOTH orders.
        source_nodes = [
            {"id": "a", "entity_id": "a"},
            {"id": "b", "entity_id": "b"},
        ]
        target_nodes = [dict(n) for n in source_nodes]
        target_edges = [_edge("a", "b", weight=2)]
        parallel = [_edge("a", "b", weight=1), _edge("a", "b", weight=2)]
        for source_edges in (parallel, list(reversed(parallel))):
            report = verify_source_side(
                source_nodes, source_edges, target_nodes, target_edges
            )
            assert not report.ok
            assert report.source_parallel_violations == {("a", "b"): 2}

    def test_report_deterministic_across_enumeration_orders(self):
        # The full report (every field, not just ok) must be independent of
        # the source enumeration order, including on invariant-violating
        # input where last-write-wins map construction could otherwise vary.
        source_nodes = [
            {"id": "a", "entity_id": "a"},
            {"id": "b", "entity_id": "b"},
            {"id": "c", "entity_id": "c"},
        ]
        source_edges = [
            _edge("a", "b", weight=1),
            _edge("a", "b", weight=2),  # parallel violation
            _edge("b", "c", weight=3),
            _edge("c", "b", weight=4),  # divergent reciprocal
        ]
        target_nodes = [dict(n) for n in source_nodes]
        target_edges = [_edge("a", "b", weight=2), _edge("b", "c", weight=3)]
        report_fwd = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        report_rev = verify_source_side(
            list(reversed(source_nodes)),
            list(reversed(source_edges)),
            list(reversed(target_nodes)),
            list(reversed(target_edges)),
        )
        assert report_fwd == report_rev
        assert not report_fwd.ok

    @pytest.mark.parametrize(
        "violation",
        ["duplicate_node", "equal_reciprocal"],
    )
    def test_source_invariant_violation_reported_even_with_equal_payload(
        self, violation
    ):
        """Verify runs the same fail-closed detectors as the plan phase, even
        when duplicate rows or reciprocals carry byte-equal payloads."""
        source_nodes, source_edges, target_nodes, target_edges = self._clean_migration()
        if violation == "duplicate_node":
            source_nodes = source_nodes + [dict(source_nodes[0])]
        else:
            source_edges = source_edges + [_edge("b", "a", weight=1.0)]

        report = verify_source_side(
            source_nodes, source_edges, target_nodes, target_edges
        )
        assert not report.ok
        if violation == "duplicate_node":
            assert report.source_duplicate_node_ids == ["a"]
        else:
            assert report.source_reciprocal_pairs == [("a", "b")]


# ---------------------------------------------------------------------------
# End-to-end determinism of the pure pipeline (AC-c)
# ---------------------------------------------------------------------------


class TestDeterministicPipeline:
    def test_canonical_write_set_identical_across_two_runs(self):
        # AC-c second half: two runs over differently enumerated source reads
        # produce identical canonical write sets. This exercises the ordering
        # pipeline only — the fail-closed gates (reciprocals, duplicates,
        # parallel rows) are the callers' job and are covered separately, so
        # this input deliberately includes a reciprocal pair to prove that
        # canonicalization itself is enumeration-order independent.
        run_a = [
            _edge("b", "a", weight=1.0),
            _edge("a", "b", weight=1.0),
            _edge('He said "hi"', "O'Brien", rel="quote"),
            _edge("한국", "일본", rel="이웃"),
            _edge("loop", "loop", rel="self"),
        ]
        run_b = list(reversed(run_a))

        def pipeline(directed_edges):
            assert not any(c > 1 for c in count_parallel_edges(directed_edges).values())
            ordered = sort_directed_edges(directed_edges)
            return [
                (
                    *canonicalize_edge(e["source"], e["target"]),
                    tuple(sorted(edge_properties(e).items())),
                )
                for e in ordered
            ]

        assert pipeline(run_a) == pipeline(run_b)

    def test_edge_properties_strips_only_endpoints(self):
        e = _edge("a", "b", source_id="chunk-1", weight=1)
        assert edge_properties(e) == {"source_id": "chunk-1", "weight": 1}


# ---------------------------------------------------------------------------
# Async orchestration — fakes standing in for the storage backends
# ---------------------------------------------------------------------------


class _FakeAGESource:
    """AGE-shaped source: returns whatever node/edge dicts it was seeded with."""

    def __init__(self, nodes, edges, workspace=""):
        self._nodes = nodes
        self._edges = edges
        # Real graph storages carry the workspace they resolved to; the tool
        # cross-checks it against --workspace before doing anything
        # destructive, so the fake must carry one too.
        self.workspace = workspace
        self.finalized = False

    async def initialize(self):
        pass

    async def finalize(self):
        self.finalized = True

    async def get_all_nodes(self):
        return [dict(n) for n in self._nodes]

    async def get_all_edges(self):
        return [dict(e) for e in self._edges]


class PGGraphStorage(_FakeAGESource):
    """Named to satisfy the class-name allowlist; behavior is the fake's."""


class PGTableGraphStorage:
    """Fake target mirroring the pgtable contracts the tool relies on:

    - upsert_edges_batch canonicalizes via Python min/max and silently
      auto-creates missing endpoints as stub nodes (returns None);
    - remove_* mirror DELETE ... = ANY: absent ids are silent no-ops;
    - get_all_* merge ids into the dicts ("id" / "source"+"target");
    - is_empty records the call and returns None. Real graph storages do
      not have this attribute AT ALL (the None-returning stub in base.py is
      BaseKVStorage's; BaseGraphStorage defines nothing like it) — the fake
      carries one purely as a never-call tripwire so AC-a can assert the
      gate never even attempts it.

    ``foreign_nodes`` stands in for another workspace slice: the real
    remove_* are workspace-scoped, so compensation must leave it untouched.
    """

    def __init__(
        self,
        seed_nodes=None,
        foreign_nodes=None,
        fail_edge_batch_after=None,
        drop_result=_UNSET,
        workspace="",
    ):
        # drop_result: force drop() to report a failure the way the real
        # backend does (a returned {"status": "error"} dict, not an exception).
        self.drop_result = drop_result
        self.workspace = workspace
        self.nodes: dict[str, dict[str, Any]] = dict(seed_nodes or {})
        self.edges: dict[tuple[str, str], dict[str, Any]] = {}
        self.foreign_nodes: dict[str, dict[str, Any]] = dict(foreign_nodes or {})
        self.calls: list[str] = []
        self.remove_edges_args = None
        self.remove_nodes_args = None
        self.is_empty_called = False
        self.fail_edge_batch_after = fail_edge_batch_after
        self.finalized = False

    async def initialize(self):
        pass

    async def finalize(self):
        self.finalized = True

    async def is_empty(self):
        self.is_empty_called = True  # tripwire: real graph storages lack this

    async def get_all_nodes(self):
        self.calls.append("get_all_nodes")
        return sorted(
            ({**props, "id": nid} for nid, props in self.nodes.items()),
            key=lambda n: n["id"],
        )

    async def get_all_edges(self):
        self.calls.append("get_all_edges")
        return [
            {**props, "source": src, "target": tgt}
            for (src, tgt), props in sorted(self.edges.items())
        ]

    async def upsert_nodes_batch(self, nodes):
        self.calls.append("upsert_nodes_batch")
        for node_id, props in nodes:
            assert "entity_id" in props
            self.nodes[node_id] = {**self.nodes.get(node_id, {}), **props}

    async def upsert_edges_batch(self, edges):
        self.calls.append("upsert_edges_batch")
        for i, (src, tgt, props) in enumerate(edges):
            if (
                self.fail_edge_batch_after is not None
                and i >= self.fail_edge_batch_after
            ):
                raise RuntimeError("injected edge-batch failure")
            key = (min(src, tgt), max(src, tgt))
            for nid in key:
                # Silent endpoint auto-creation (CTE ON CONFLICT DO NOTHING).
                self.nodes.setdefault(nid, {"entity_id": nid})
            self.edges[key] = dict(props)

    async def remove_edges(self, edges):
        self.calls.append("remove_edges")
        self.remove_edges_args = list(edges)
        for src, tgt in edges:
            self.edges.pop((min(src, tgt), max(src, tgt)), None)  # absent: no-op

    async def remove_nodes(self, nodes):
        self.calls.append("remove_nodes")
        self.remove_nodes_args = list(nodes)
        for nid in nodes:
            self.nodes.pop(nid, None)  # absent: no-op

    async def drop(self):
        self.calls.append("drop")
        # Mirrors the real contract: PGTableGraphStorage.drop() reports failure
        # by RETURNING {"status": "error", ...} rather than raising, so a fake
        # that returned None (or always succeeded) would hide an unchecked
        # return value in the caller.
        if self.drop_result is not _UNSET:
            return self.drop_result
        self.nodes.clear()
        self.edges.clear()
        return {"status": "success", "message": "data dropped"}


def _source(nodes=None, edges=None):
    node_dicts = [
        {"id": nid, "entity_id": nid, "entity_type": "thing"} for nid in (nodes or [])
    ]
    return PGGraphStorage(node_dicts, edges or [])


class TestSourceNodeIdentity:
    """A real AGE graph can hold a vertex with no properties (`CREATE (n:base)`),
    which enumerates as {"id": None}. That must be refused, not crashed on."""

    @pytest.mark.parametrize(
        "nodes, match, force, seed",
        [
            pytest.param([{"id": None}], "non-string id", False, None, id="none_id"),
            pytest.param(
                [{"id": "alice", "entity_id": "alice"}, {"id": None}],
                None,
                False,
                None,
                id="mixed_valid_and_none",
            ),
            pytest.param(
                [{"id": "a", "entity_id": "b"}],
                "disagrees",
                False,
                None,
                id="entity_id_disagrees",
            ),
            pytest.param(
                [{"id": None}],
                None,
                True,
                {"pre": {"entity_id": "pre"}},
                id="force_mode_refuses_before_drop",
            ),
        ],
    )
    async def test_invalid_source_node_identity_is_refused(
        self, nodes, match, force, seed
    ):
        target = PGTableGraphStorage(seed_nodes=seed)
        source = PGGraphStorage(nodes, [])
        with pytest.raises(MigrationDataError, match=match):
            await migrate_graph(source, target, force_empty_target=force)
        assert target.calls.count("upsert_nodes_batch") == 0
        if force:
            assert "drop" not in target.calls
            assert target.nodes == seed


class TestDetectSelfLoopEdges:
    """The migration tool refuses a source graph that contains a self-loop.

    ``BaseGraphStorage.node_degree`` states a graph must not hold one, so the
    source is invariant-violating. Neither alternative to refusing is safe:
    carrying it across propagates the violation, and DROPPING it orphans the
    relation's ``relationships_vdb`` row -- this tool migrates the graph only,
    and ``adelete_by_relation`` 404s once the graph edge is gone, so the row
    becomes unreachable through every public API. Refusing keeps the cleanup
    possible at the source, where the graph edge still exists.
    """

    def test_reports_every_offender_sorted_and_deduped(self):
        """The message names what the operator has to clean up, so the list
        must be complete and stable."""
        edges = [
            {"source": "z", "target": "z"},
            {"source": "a", "target": "a"},
            {"source": "z", "target": "z"},
        ]

        assert detect_self_loop_edges(edges) == ["a", "z"]

    def test_reciprocals_are_not_self_loops(self):
        """Only ``src == tgt``: a reciprocal pair is a different refusal, and
        folding it in here would hide what detect_reciprocal_pairs reports."""
        edges = [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]

        assert detect_self_loop_edges(edges) == []
        assert detect_reciprocal_pairs(edges) == [("a", "b")]


class TestCardinalityPreservingPolicy:
    """Reciprocals and duplicates are refused regardless of payload equality:
    collapsing them preserves payloads but changes degree and cardinality."""

    async def test_equal_reciprocal_edges_are_refused(self):
        target = PGTableGraphStorage()
        source = _source(
            ["a", "b"], [_edge("a", "b", weight=1.0), _edge("b", "a", weight=1.0)]
        )
        with pytest.raises(MigrationDataError, match="reciprocal"):
            await migrate_graph(source, target)
        assert target.calls.count("upsert_nodes_batch") == 0

    def test_duplicate_detection_ignores_payloads(self):
        nodes = [{"id": "a", "entity_id": "a", "p": 1}, {"id": "a", "entity_id": "a"}]
        assert detect_duplicate_node_ids(nodes) == ["a"]


class TestJsonbRepresentability:
    """agtype represents values PostgreSQL jsonb cannot: NaN/+-Infinity, -0.0
    (whose sign is normalised away), and strings holding a NUL or an unpaired
    surrogate (which jsonb's underlying text type cannot hold)."""

    async def test_non_finite_payload_is_refused_before_the_drop(self):
        target = PGTableGraphStorage(seed_nodes={"pre": {"entity_id": "pre"}})
        source = _source(["a", "b"], [_edge("a", "b", weight=float("nan"))])
        with pytest.raises(MigrationDataError, match="NaN"):
            await migrate_graph(source, target, force_empty_target=True)
        # The point of the check: the destructive step never ran.
        assert "drop" not in target.calls
        assert target.nodes == {"pre": {"entity_id": "pre"}}

    @pytest.mark.parametrize(
        "records, expected",
        [
            pytest.param(
                [
                    ("edge x", {"w": float("inf")}),
                    ("node y", {"meta": {"vals": [1.0, float("-inf")]}}),
                    ("node z", {"w": 1.0}),
                ],
                [("edge x", ["w"]), ("node y", ["meta.vals[1]"])],
                id="nan_and_infinities",
            ),
            pytest.param(
                [
                    ("node nul", {"description": "before\x00after"}),
                    ("edge surrogate", {"meta": {"vals": ["ok", "lone\ud800"]}}),
                ],
                [("edge surrogate", ["meta.vals[1]"]), ("node nul", ["description"])],
                id="nul_and_lone_surrogate",
            ),
            pytest.param(
                [("node k", {"bad\x00key": 1})],
                [("node k", ["bad\x00key"])],
                id="offending_key",
            ),
        ],
    )
    def test_finds_non_jsonb_representable_values_and_keys(self, records, expected):
        assert find_non_jsonb_representable(records) == expected

    def test_ordinary_unicode_passes(self):
        # The gate must not become a general non-ASCII filter: everything here
        # round-trips through jsonb unchanged. "\U0001f600" is a PAIRED
        # surrogate pair on narrow builds and a single astral code point here;
        # either way it is valid UTF-8 and must not be flagged.
        assert (
            find_non_jsonb_representable(
                [
                    ("node ko", {"entity_id": "한국", "desc": "이웃 관계"}),
                    ("node astral", {"label": "graph \U0001f600 store"}),
                    ("node quotes", {"q": 'He said "hi" - O\'Brien'}),
                ]
            )
            == []
        )


class TestNegativeZero:
    def test_signed_zero_is_a_difference(self):
        assert not payloads_equal(-0.0, 0.0)
        assert payloads_equal(-0.0, -0.0)
        assert not payloads_equal({"w": -0.0}, {"w": 0.0})


class TestDropStatusContract:
    @pytest.mark.parametrize(
        "drop_result",
        # A non-success status is TestForceModeOrdering's regression case.
        [None, {"message": "no status"}, "success"],
        ids=["none", "no-status", "not-a-dict"],
    )
    async def test_only_explicit_success_counts_as_dropped(self, drop_result):
        # "not an error" is not the same as "succeeded": a no-op drop returning
        # any of these would otherwise let the run write into rows it does not
        # own and then compensate against them.
        target = PGTableGraphStorage(
            seed_nodes={"pre": {"entity_id": "pre"}}, drop_result=drop_result
        )
        source = _source(["a", "b"], [_edge("a", "b", weight=1.0)])
        with pytest.raises(MigrationPreconditionError, match="status"):
            await migrate_graph(source, target, force_empty_target=True)
        assert target.calls.count("upsert_nodes_batch") == 0
        assert target.nodes == {"pre": {"entity_id": "pre"}}


class TestCancellationCompensates:
    async def test_cancellation_mid_write_still_compensates(self):
        # Regression: the write block caught `Exception`, but
        # asyncio.CancelledError does not subclass it — a Ctrl-C or a cancelled
        # task partway through the writes left the target populated with no
        # compensation and no report, which is the exact outcome the id-scoped
        # undo exists to prevent.
        target = PGTableGraphStorage()

        async def cancel_during_edges(edges):
            target.calls.append("upsert_edges_batch")
            raise asyncio.CancelledError()

        target.upsert_edges_batch = cancel_during_edges
        source = _source(["a", "b"], [_edge("a", "b", weight=1.0)])

        with pytest.raises(asyncio.CancelledError):
            await migrate_graph(source, target)

        # Compensation ran, and cancellation kept propagating as itself.
        assert "remove_edges" in target.calls
        assert "remove_nodes" in target.calls
        assert target.nodes == {}


class TestForceModeOrdering:
    """force_empty_target drops pre-existing data; that must be the LAST thing
    that can go wrong, not the first thing that happens."""

    async def test_source_check_failure_does_not_destroy_the_target(self):
        # Regression: drop() used to run before the source was validated, so a
        # source that fails a fail-closed check cost the operator their
        # existing target slice and produced nothing in return.
        target = PGTableGraphStorage(seed_nodes={"pre": {"entity_id": "pre"}})
        source = PGGraphStorage(
            [
                {"id": "a", "entity_id": "a", "description": "one"},
                {"id": "a", "entity_id": "a", "description": "two"},
            ],
            [],
        )
        with pytest.raises(MigrationDataError):
            await migrate_graph(source, target, force_empty_target=True)
        assert "drop" not in target.calls, "the target was dropped before validation"
        assert target.nodes == {"pre": {"entity_id": "pre"}}

    async def test_failed_drop_is_not_reported_as_a_successful_drop(self):
        # Regression: PGTableGraphStorage.drop() reports failure by RETURNING
        # {"status": "error"} rather than raising. An unchecked return value
        # recorded the slice as dropped and let the migration write into — and
        # later compensate against — rows it does not own.
        target = PGTableGraphStorage(
            seed_nodes={"pre": {"entity_id": "pre"}},
            drop_result={"status": "error", "message": "permission denied"},
        )
        source = _source(["a", "b"], [_edge("a", "b", weight=1.0)])
        with pytest.raises(MigrationPreconditionError, match="permission denied"):
            await migrate_graph(source, target, force_empty_target=True)
        assert "upsert_nodes_batch" not in target.calls
        assert target.nodes == {"pre": {"entity_id": "pre"}}


class TestResolvedWorkspace:
    """POSTGRES_WORKSPACE outranks the constructor argument, so --workspace
    alone does not determine which slice a destructive run acts on."""

    @pytest.mark.parametrize(
        "requested, src_ws, tgt_ws, match",
        [
            pytest.param(
                "staging", "production", "production", "was not named", id="env_override"
            ),
            pytest.param(
                "", "a", "b", "different workspaces", id="divergent_backends"
            ),
        ],
    )
    def test_check_resolved_workspace_refuses_mismatch(
        self, requested, src_ws, tgt_ws, match
    ):
        source = PGGraphStorage([], [])
        source.workspace = src_ws
        target = PGTableGraphStorage(workspace=tgt_ws)
        with pytest.raises(MigrationPreconditionError, match=match):
            _check_resolved_workspace(requested, source, target)

    def test_empty_request_accepts_whatever_the_backends_resolve(self):
        source = PGGraphStorage([], [])
        source.workspace = "default"
        target = PGTableGraphStorage(workspace="default")
        _check_resolved_workspace("", source, target)

    @pytest.mark.parametrize(
        "env_ws, ini_content, match, forbidden",
        [
            pytest.param("production", None, "outranks it", None, id="env_override"),
            pytest.param(
                None,
                "[postgres]\nworkspace = production\n",
                "config.ini",
                None,
                id="config_ini_override",
            ),
            pytest.param(
                "production",
                "[postgres]\nworkspace = fromfile\n",
                "POSTGRES_WORKSPACE",
                "fromfile",
                id="env_outranks_config_ini",
            ),
        ],
    )
    def test_precheck_refuses_configured_workspace_override(
        self, monkeypatch, tmp_path, env_ws, ini_content, match, forbidden
    ):
        """POSTGRES_WORKSPACE and config.ini [postgres] workspace overrides are
        refused before any connection opens, with env outranking config.ini."""
        if ini_content is not None:
            (tmp_path / "config.ini").write_text(ini_content, encoding="utf-8")
            monkeypatch.chdir(tmp_path)
        if env_ws is None:
            monkeypatch.delenv("POSTGRES_WORKSPACE", raising=False)
        else:
            monkeypatch.setenv("POSTGRES_WORKSPACE", env_ws)

        with pytest.raises(MigrationPreconditionError, match=match) as excinfo:
            _precheck_requested_workspace("staging")
        assert "production" in str(excinfo.value)
        if forbidden is not None:
            assert forbidden not in str(excinfo.value)
        # Matching or unnamed requests still pass through the same branch.
        _precheck_requested_workspace("production")
        _precheck_requested_workspace("")

    @pytest.mark.parametrize(
        "env_ws, ini_content, allowed_requests",
        [
            pytest.param(
                None,
                "[postgres]\nhost = localhost\n",
                ("staging",),
                id="config_ini_without_workspace_key",
            ),
            pytest.param(
                "production",
                None,
                ("production", ""),
                id="matching_or_unnamed_under_env",
            ),
            pytest.param(None, None, ("staging",), id="no_override_set"),
        ],
    )
    def test_precheck_allows_compatible_workspace(
        self, monkeypatch, tmp_path, env_ws, ini_content, allowed_requests
    ):
        if ini_content is not None:
            (tmp_path / "config.ini").write_text(ini_content, encoding="utf-8")
            monkeypatch.chdir(tmp_path)
        if env_ws is None:
            monkeypatch.delenv("POSTGRES_WORKSPACE", raising=False)
        else:
            monkeypatch.setenv("POSTGRES_WORKSPACE", env_ws)
        for req in allowed_requests:
            _precheck_requested_workspace(req)

    def test_cli_refuses_env_override_before_touching_the_factory(
        self, monkeypatch, capsys
    ):
        import lightrag.kg.factory as factory

        def exploding_get_storage_class(name):
            raise AssertionError(
                f"no backend may be constructed for a refused workspace (got {name!r})"
            )

        monkeypatch.setattr(factory, "get_storage_class", exploding_get_storage_class)
        monkeypatch.setenv("POSTGRES_WORKSPACE", "production")
        with pytest.raises(SystemExit) as excinfo:
            main(["--workspace", "staging"])
        assert excinfo.value.code == 1
        assert "outranks it" in capsys.readouterr().out


class TestMigrationPreconditions:
    async def test_non_empty_target_rejected_before_any_write(self):
        # AC-a: rejection via get_all_nodes, is_empty never awaited, non-zero
        # exit contract, zero writes issued.
        target = PGTableGraphStorage(seed_nodes={"pre": {"entity_id": "pre"}})
        with pytest.raises(MigrationPreconditionError) as excinfo:
            await migrate_graph(_source(["a"], [_edge("a", "a")]), target)
        assert excinfo.value.exit_code != 0
        assert target.is_empty_called is False
        assert target.calls == ["get_all_nodes"]  # the gate read; no writes
        assert target.nodes == {"pre": {"entity_id": "pre"}}

    async def test_pair_allowlist_rejects_unknown_backends(self):
        class SomeOtherStorage(_FakeAGESource):
            pass

        target = PGTableGraphStorage()
        with pytest.raises(MigrationPreconditionError):
            await migrate_graph(SomeOtherStorage([], []), target)
        assert target.calls == []  # rejected before even the empty gate

        class SomeOtherTarget(PGTableGraphStorage):
            pass

        with pytest.raises(MigrationPreconditionError):
            await migrate_graph(_source(), SomeOtherTarget())

    async def test_target_slice_is_empty_uses_get_all_nodes(self):
        target = PGTableGraphStorage()
        assert await target_slice_is_empty(target) is True
        target.nodes["x"] = {"entity_id": "x"}
        assert await target_slice_is_empty(target) is False
        assert target.is_empty_called is False

    async def test_force_empty_target_drops_then_migrates(self):
        target = PGTableGraphStorage(seed_nodes={"pre": {"entity_id": "pre"}})
        report = await migrate_graph(
            _source(["a", "b"], [_edge("a", "b", weight=1.0)]),
            target,
            force_empty_target=True,
        )
        assert "drop" in target.calls
        assert report.verified
        assert "pre" not in target.nodes

    @pytest.mark.parametrize(
        "nodes, edges",
        [
            pytest.param(
                [
                    {"id": "a", "entity_id": "a", "entity_type": "thing"},
                    {"id": "b", "entity_id": "b", "entity_type": "thing"},
                ],
                [_edge("a", "b", w=1), _edge("a", "b", w=2)],
                id="parallel_edges",
            ),
            pytest.param(
                [
                    {"id": "a", "entity_id": "a", "p": 1},
                    {"id": "a", "entity_id": "a", "p": 2},
                ],
                [],
                id="duplicate_divergent_nodes",
            ),
            pytest.param(
                [{"id": "a", "entity_type": "thing"}],
                [],
                id="missing_entity_id",
            ),
        ],
    )
    async def test_invalid_source_data_fails_before_any_write(self, nodes, edges):
        target = PGTableGraphStorage()
        source = PGGraphStorage(nodes, edges)
        with pytest.raises(MigrationDataError):
            await migrate_graph(source, target)
        assert "upsert_nodes_batch" not in target.calls
        assert "upsert_edges_batch" not in target.calls
        assert target.nodes == {}


class TestMigrationSuccess:
    async def test_full_migration_verified(self):
        source = _source(
            ["a", "b", "한국"],
            [
                _edge("a", "b", weight=1.0),
                _edge("한국", "a", rel="이웃"),
            ],
        )
        target = PGTableGraphStorage()
        report = await migrate_graph(source, target)
        assert report.verified
        assert report.verification is not None and report.verification.ok
        assert report.node_count == 3
        assert report.edge_count == 2  # reciprocal deduped
        assert set(target.edges) == {("a", "b"), ("a", "한국")}
        assert report.compensated is False

    async def test_auto_created_endpoint_without_source_node_fails_closed(self):
        # "ghost" appears only as an edge endpoint. A real AGE source cannot
        # produce this (get_all_edges joins endpoints to the node table), so
        # if it happens the enumeration was incomplete: the target
        # auto-creates a stub node the source never enumerated, source-driven
        # verification flags it as unexpected, and the run compensates. The
        # derived write set must carry the stub so compensation removes it.
        source = _source(["a"], [_edge("a", "ghost", w=1)])
        target = PGTableGraphStorage()
        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.written_node_ids == ["a", "ghost"]
        assert report.verification is not None
        assert report.verification.unexpected_node_ids == ["ghost"]
        assert report.compensated
        assert target.nodes == {} and target.edges == {}

    async def test_verification_fails_when_target_drops_an_edge(self):
        class LossyTarget(PGTableGraphStorage):
            async def upsert_edges_batch(self, edges):
                await super().upsert_edges_batch(edges[:-1])  # silently drop one

        # Keep the class-name allowlist matching despite the subclass.
        LossyTarget.__name__ = "PGTableGraphStorage"

        source = _source(["a", "b", "c"], [_edge("a", "b", w=1), _edge("b", "c", w=2)])
        target = LossyTarget()
        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.verification is not None and not report.verification.ok
        assert report.verification.missing_edges == [("b", "c")]
        assert report.compensated
        assert target.nodes == {} and target.edges == {}


class TestMigrationCompensation:
    async def test_edge_batch_failure_compensates_exactly_written_ids(self):
        # AC-b: injected edge-batch failure -> compensation removes exactly
        # the derived written set (auto-created endpoints included), and
        # data outside that set (another workspace slice) is untouched.
        source = _source(["a", "b"], [_edge("a", "b", w=1), _edge("b", "ghost", w=2)])
        target = PGTableGraphStorage(
            foreign_nodes={"other": {"entity_id": "other"}},
            fail_edge_batch_after=1,  # first edge applies, second raises
        )
        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.compensated
        assert report.compensation_error is None
        # Exactly the derived superset was passed — including "ghost", whose
        # edge never landed (absent-id no-op contract), and nothing more.
        assert report.written_node_ids == ["a", "b", "ghost"]
        assert target.remove_nodes_args == ["a", "b", "ghost"]
        assert target.remove_edges_args == [("a", "b"), ("b", "ghost")]
        # Slice is clean again; the foreign slice was never touched.
        assert target.nodes == {} and target.edges == {}
        assert target.foreign_nodes == {"other": {"entity_id": "other"}}

    async def test_compensation_removes_edges_before_nodes(self):
        # FK order is asserted, not assumed.
        source = _source(["a", "b"], [_edge("a", "b", w=1)])
        target = PGTableGraphStorage(fail_edge_batch_after=0)
        with pytest.raises(MigrationWriteError):
            await migrate_graph(source, target)
        assert target.calls.index("remove_edges") < target.calls.index("remove_nodes")

    async def test_absent_ids_are_noops_for_compensation(self):
        # Pin the DELETE ... = ANY contract the tool relies on: compensating
        # with ids that were never written must not raise and must not touch
        # unrelated rows.
        target = PGTableGraphStorage(seed_nodes={"keep": {"entity_id": "keep"}})
        await compensate_partial_migration(
            target, ["never-written"], [("never", "written")]
        )
        assert target.nodes == {"keep": {"entity_id": "keep"}}

    async def test_failed_compensation_is_reported_not_swallowed(self):
        class BrokenRemoveTarget(PGTableGraphStorage):
            # Parameter name matches the base signature (keyword callers must
            # keep working); it is unused because the fake fails up front.
            async def remove_nodes(self, nodes):
                del nodes
                raise RuntimeError("connection lost during compensation")

        # Keep the class-name allowlist matching despite the subclass.
        BrokenRemoveTarget.__name__ = "PGTableGraphStorage"

        source = _source(["a", "b"], [_edge("a", "b", w=1)])
        target = BrokenRemoveTarget(fail_edge_batch_after=0)
        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.compensated is False
        assert "connection lost" in report.compensation_error
        # Manual-cleanup contract: the report still carries the full sets.
        assert report.written_node_ids == ["a", "b"]
        assert report.written_edge_keys == [("a", "b")]
        assert "COMPENSATION INCOMPLETE" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI — argparse wiring, exit codes, factory-based instantiation
# ---------------------------------------------------------------------------


def _patch_factory(monkeypatch, source, target):
    """Route the CLI's lazy get_storage_class import to the prepared fakes.

    Returns the constructor kwargs recorded per backend name so tests can
    assert what the CLI instantiated with. Never touches a real database.
    """
    import lightrag.kg.factory as factory

    ctor_kwargs: dict[str, dict[str, Any]] = {}
    instances = {"PGGraphStorage": source, "PGTableGraphStorage": target}

    def fake_get_storage_class(name):
        instance = instances[name]

        def ctor(**kwargs):
            ctor_kwargs[name] = kwargs
            # Mimic the backends resolving their workspace. Without a
            # POSTGRES_WORKSPACE override they land on what was requested,
            # which is what the tool cross-checks before acting.
            instance.workspace = kwargs.get("workspace", "")
            return instance

        return ctor

    monkeypatch.setattr(factory, "get_storage_class", fake_get_storage_class)
    return ctor_kwargs


class TestCli:
    def _fakes(self):
        source = _source(["a", "b"], [_edge("a", "b", weight=1.0)])
        target = PGTableGraphStorage()
        return source, target

    def test_dry_run_default_writes_nothing(self, monkeypatch, capsys):
        # AC-1: the default invocation is a read-only preflight — zero write
        # calls of any kind on the target, and a report is still produced.
        source, target = self._fakes()
        _patch_factory(monkeypatch, source, target)
        main([])  # no SystemExit: dry run succeeded
        assert set(target.calls) <= {"get_all_nodes", "get_all_edges"}
        for write_call in (
            "upsert_nodes_batch",
            "upsert_edges_batch",
            "remove_nodes",
            "remove_edges",
            "drop",
        ):
            assert write_call not in target.calls
        assert target.nodes == {} and target.edges == {}
        out = capsys.readouterr().out
        assert "'mode': 'dry-run'" in out
        assert "'nodes': 2" in out and "'edges': 1" in out
        assert "'verified': False" in out
        # Empty target: no destructive consequence to disclose.
        assert "'target_non_empty': False" in out
        assert "'would_drop_target_slice': False" in out
        assert source.finalized and target.finalized

    def test_apply_performs_migration_and_exits_zero(self, monkeypatch, capsys):
        source, target = self._fakes()
        _patch_factory(monkeypatch, source, target)
        main(["--apply"])  # no SystemExit: exit code zero
        assert target.edges == {("a", "b"): {"weight": 1.0}}
        out = capsys.readouterr().out
        assert "'mode': 'apply'" in out
        assert "'verified': True" in out
        assert "'dropped_target_slice': False" in out

    @pytest.mark.parametrize(
        "fault, expected_snippet",
        [
            pytest.param("non_empty_target", "'error':", id="precondition_failure"),
            pytest.param("edge_batch_write", "'compensated': True", id="write_failure"),
        ],
    )
    def test_cli_failure_exits_nonzero_and_reports(
        self, monkeypatch, capsys, fault, expected_snippet
    ):
        source, target = self._fakes()
        if fault == "non_empty_target":
            target.nodes["pre"] = {"entity_id": "pre"}
        else:
            target.fail_edge_batch_after = 0
        _patch_factory(monkeypatch, source, target)
        with pytest.raises(SystemExit) as excinfo:
            main(["--apply"])
        assert excinfo.value.code == 1
        assert expected_snippet in capsys.readouterr().out
        if fault == "non_empty_target":
            assert "upsert_nodes_batch" not in target.calls
            assert target.nodes == {"pre": {"entity_id": "pre"}}
        assert source.finalized and target.finalized

    def test_rejected_pair_exits_nonzero_before_instantiation(
        self, monkeypatch, capsys
    ):
        import lightrag.kg.factory as factory

        def exploding_get_storage_class(name):
            raise AssertionError(
                f"factory must not be called for a rejected pair (got {name!r})"
            )

        monkeypatch.setattr(factory, "get_storage_class", exploding_get_storage_class)
        with pytest.raises(SystemExit) as excinfo:
            main(["--source-backend", "NetworkXStorage"])
        assert excinfo.value.code == 1
        assert "unsupported migration pair" in capsys.readouterr().out

    def test_shared_data_is_initialized_before_any_storage_is_built(self, monkeypatch):
        """Regression: the CLI must call initialize_share_data() first."""
        import lightrag.kg.shared_storage as shared_storage

        order: list[str] = []

        def recording_initialize_share_data(*args, **kwargs):
            order.append(f"initialize_share_data{args or ()}")

        monkeypatch.setattr(
            shared_storage, "initialize_share_data", recording_initialize_share_data
        )

        source, target = self._fakes()
        import lightrag.kg.factory as factory

        instances = {"PGGraphStorage": source, "PGTableGraphStorage": target}

        def fake_get_storage_class(name):
            def ctor(**kwargs):
                order.append(f"build:{name}")
                return instances[name]

            return ctor

        monkeypatch.setattr(factory, "get_storage_class", fake_get_storage_class)

        main([])

        assert order, "initialize_share_data was never called"
        assert order[0].startswith("initialize_share_data"), (
            f"shared data must be initialized before any storage is built, got {order}"
        )
        assert "build:PGGraphStorage" in order

    @pytest.mark.parametrize(
        "argv, expected_ws",
        [
            pytest.param(["--workspace", "cli-ws"], "cli-ws", id="flag_overrides_env"),
            pytest.param([], "env-ws", id="defaults_to_env"),
        ],
    )
    def test_workspace_flag_and_env_resolution(self, monkeypatch, argv, expected_ws):
        source, target = self._fakes()
        ctor_kwargs = _patch_factory(monkeypatch, source, target)
        monkeypatch.setenv("WORKSPACE", "env-ws")
        main(argv)
        assert ctor_kwargs["PGGraphStorage"]["workspace"] == expected_ws
        assert ctor_kwargs["PGTableGraphStorage"]["workspace"] == expected_ws

    @pytest.mark.parametrize(
        "argv, expect_drop, expected_snippets",
        [
            pytest.param(
                ["--force-empty-target"],
                False,
                (
                    "'mode': 'dry-run'",
                    "'target_non_empty': True",
                    "'would_drop_target_slice': True",
                ),
                id="dry_run_never_drops",
            ),
            pytest.param(
                ["--apply", "--force-empty-target"],
                True,
                ("'target_non_empty': True", "'dropped_target_slice': True"),
                id="apply_drops_and_reports",
            ),
        ],
    )
    def test_force_empty_target_dry_run_vs_apply(
        self, monkeypatch, capsys, argv, expect_drop, expected_snippets
    ):
        source, target = self._fakes()
        target.nodes["pre"] = {"entity_id": "pre"}
        _patch_factory(monkeypatch, source, target)
        main(argv)
        assert ("drop" in target.calls) is expect_drop
        if not expect_drop:
            assert target.nodes["pre"] == {"entity_id": "pre"}
        out = capsys.readouterr().out
        for snippet in expected_snippets:
            assert snippet in out

    def test_storage_init_failure_reports_and_finalizes(self, monkeypatch, capsys):
        source, _ = self._fakes()

        class FailingInitTarget(PGTableGraphStorage):
            async def initialize(self):
                raise RuntimeError("db unreachable")

        target = FailingInitTarget()
        _patch_factory(monkeypatch, source, target)
        with pytest.raises(SystemExit) as excinfo:
            main(["--apply"])
        assert excinfo.value.code == 1
        assert "storage initialization failed" in capsys.readouterr().out
        assert source.finalized
        assert target.finalized

    async def test_dry_run_migration_rejects_non_empty_target(self):
        source, target = self._fakes()
        target.nodes["pre"] = {"entity_id": "pre"}
        with pytest.raises(MigrationPreconditionError):
            await dry_run_migration(source, target)

    def test_cli_import_is_offline_safe(self):
        import lightrag

        repo_root = Path(lightrag.__file__).resolve().parents[1]
        code = (
            "import sys\n"
            "import lightrag.tools.migrate_graph_storage as m\n"
            "assert callable(m.main)\n"
            "assert 'asyncpg' not in sys.modules, 'CLI import pulled asyncpg'\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, cwd=repo_root)


# ---------------------------------------------------------------------------
# Integration tests against real NetworkXStorage
# ---------------------------------------------------------------------------

_NX_NODES: dict[str, dict[str, str]] = {
    "alice": {"entity_id": "alice", "entity_type": "person", "description": "plain"},
    'He said "hi"': {"entity_id": 'He said "hi"', "entity_type": "quote"},
    "한국": {"entity_id": "한국", "entity_type": "국가", "description": "유니코드"},
    "loop": {"entity_id": "loop", "entity_type": "thing"},
}
_NX_EDGES: list[tuple[str, str, dict[str, str]]] = [
    ("한국", "alice", {"weight": "1.0", "description": "unicode endpoint"}),
    ('He said "hi"', "alice", {"weight": "2.0", "source_id": "chunk-1"}),
]


@pytest.fixture
def allow_networkx_pair(monkeypatch):
    """Extend the pair allowlist for this test only — never the shipped one."""
    import lightrag.tools.migrate_graph_storage as mgs

    monkeypatch.setattr(
        mgs,
        "ALLOWED_MIGRATION_PAIRS",
        mgs.ALLOWED_MIGRATION_PAIRS | {("NetworkXStorage", "NetworkXStorage")},
    )


async def _nx_never_embed(*_args, **_kwargs):
    raise RuntimeError("graph migration tests never embed")


async def _nx_storage(tmp_path, namespace: str):
    from lightrag.kg.networkx_impl import NetworkXStorage
    from lightrag.kg.shared_storage import initialize_share_data
    from lightrag.utils import EmbeddingFunc

    initialize_share_data()
    storage = NetworkXStorage(
        namespace=namespace,
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=EmbeddingFunc(embedding_dim=1, func=_nx_never_embed),
    )
    await storage.initialize()
    return storage


async def _nx_seeded_source(tmp_path):
    source = await _nx_storage(tmp_path, "migration_src")
    await source.upsert_nodes_batch(
        [(node_id, dict(props)) for node_id, props in _NX_NODES.items()]
    )
    await source.upsert_edges_batch(
        [(src, tgt, dict(props)) for src, tgt, props in _NX_EDGES]
    )
    return source


def _nx_node_map(nodes: list[dict]) -> dict[str, dict]:
    return {n["id"]: {k: v for k, v in n.items() if k != "id"} for n in nodes}


def _nx_canonical_edge_map(edges: list[dict]) -> dict[tuple[str, str], dict]:
    return {
        canonicalize_edge(e["source"], e["target"]): {
            k: v for k, v in e.items() if k not in ("source", "target")
        }
        for e in edges
    }


class TestRealRoundTrip:
    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_round_trip_preserves_graph(self, tmp_path):
        source = await _nx_seeded_source(tmp_path)
        target = await _nx_storage(tmp_path, "migration_tgt")

        report = await migrate_graph(source, target)

        assert report.verified
        assert report.verification is not None and report.verification.ok
        assert report.node_count == len(_NX_NODES)
        assert report.edge_count == len(_NX_EDGES)
        assert _nx_node_map(await target.get_all_nodes()) == _nx_node_map(
            await source.get_all_nodes()
        )
        assert _nx_canonical_edge_map(await target.get_all_edges()) == (
            _nx_canonical_edge_map(await source.get_all_edges())
        )
        assert set(_nx_node_map(await source.get_all_nodes())) == set(_NX_NODES)

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_self_loop_in_source_is_refused_and_writes_nothing(self, tmp_path):
        source = await _nx_seeded_source(tmp_path)
        await source.upsert_edge("loop", "loop", {"weight": "3.0"})
        target = await _nx_storage(tmp_path, "migration_tgt")

        with pytest.raises(MigrationDataError, match="self-loop"):
            await migrate_graph(source, target)

        assert await target.get_all_nodes() == []
        assert await target.get_all_edges() == []

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_non_empty_real_target_rejected(self, tmp_path):
        source = await _nx_seeded_source(tmp_path)
        target = await _nx_storage(tmp_path, "migration_tgt")
        await target.upsert_nodes_batch([("pre", {"entity_id": "pre"})])
        with pytest.raises(MigrationPreconditionError):
            await migrate_graph(source, target)
        assert set(_nx_node_map(await target.get_all_nodes())) == {"pre"}

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_attributeless_source_node_fails_closed(self, tmp_path):
        source = await _nx_storage(tmp_path, "migration_src")
        await source.upsert_nodes_batch([("a", {"entity_id": "a"})])
        await source.upsert_edges_batch([("a", "ghost", {"weight": "1"})])
        target = await _nx_storage(tmp_path, "migration_tgt")
        with pytest.raises(MigrationDataError):
            await migrate_graph(source, target)
        assert await target.get_all_nodes() == []


class TestRealCompensation:
    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_edge_batch_failure_compensates_on_real_target(
        self, tmp_path, monkeypatch
    ):
        source = await _nx_seeded_source(tmp_path)
        target = await _nx_storage(tmp_path, "migration_tgt")
        other = await _nx_storage(tmp_path, "migration_other")
        await other.upsert_nodes_batch([("keep", {"entity_id": "keep"})])

        real_upsert_edges = target.upsert_edges_batch

        async def failing_upsert_edges(edges):
            await real_upsert_edges(edges[:1])
            raise RuntimeError("injected failure after first edge")

        removed_nodes: list[str] = []
        removed_edges: list[tuple[str, str]] = []
        real_remove_nodes = target.remove_nodes
        real_remove_edges = target.remove_edges

        async def recording_remove_nodes(nodes):
            removed_nodes.extend(nodes)
            await real_remove_nodes(nodes)

        async def recording_remove_edges(edges):
            removed_edges.extend(edges)
            await real_remove_edges(edges)

        monkeypatch.setattr(target, "upsert_edges_batch", failing_upsert_edges)
        monkeypatch.setattr(target, "remove_nodes", recording_remove_nodes)
        monkeypatch.setattr(target, "remove_edges", recording_remove_edges)

        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.compensated
        assert removed_nodes == report.written_node_ids == sorted(_NX_NODES)
        assert removed_edges == report.written_edge_keys
        assert await target.get_all_nodes() == []
        assert await target.get_all_edges() == []
        assert set(_nx_node_map(await other.get_all_nodes())) == {"keep"}

    @pytest.mark.usefixtures("allow_networkx_pair")
    async def test_node_batch_failure_absent_ids_are_noops(self, tmp_path, monkeypatch):
        source = await _nx_seeded_source(tmp_path)
        target = await _nx_storage(tmp_path, "migration_tgt")

        real_upsert_nodes = target.upsert_nodes_batch

        async def failing_upsert_nodes(nodes):
            await real_upsert_nodes(nodes[:1])
            raise RuntimeError("injected failure after first node")

        monkeypatch.setattr(target, "upsert_nodes_batch", failing_upsert_nodes)

        with pytest.raises(MigrationWriteError) as excinfo:
            await migrate_graph(source, target)
        report = excinfo.value.report
        assert report.compensated
        assert report.compensation_error is None
        assert await target.get_all_nodes() == []
        assert await target.get_all_edges() == []

