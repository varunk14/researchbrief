"""Cross-source synthesis.

These tests check that the agent reasons across sources rather than stitching
summaries together. A planted contradiction must come back as CONTESTED, a
three-source agreement as CONSENSUS, a lone assertion as OUTLIER, and no claim
may be silently lost along the way.
"""
from __future__ import annotations

import json

import pytest

from helpers import FakeLLM

pytestmark = pytest.mark.synthesize


def _claims(fixture):
    from researchbrief.models import Claim

    return [Claim(**c) for c in fixture["claims"]]


def _sources(fixture):
    return fixture["sources"]


def _grouping_response(fixture) -> str:
    """The model's job is grouping. Verdicts are assigned in code."""
    return json.dumps({"clusters": [
        {"topic_label": g["topic_label"],
         "claim_ids": g["claim_ids"],
         "opposing_pairs": g["opposing_pairs"],
         "rationale": "planted fixture rationale"}
        for g in fixture["expected_grouping"]
    ]})


def test_cluster_contract():
    from researchbrief.models import ClaimCluster

    required = {
        "id", "topic_label", "claim_ids", "verdict",
        "agreeing_sources", "disagreeing_sources", "rationale",
    }
    assert required <= set(ClaimCluster.__dataclass_fields__)


def test_planted_contradiction_is_flagged_contested(synthetic_claims):
    from researchbrief.synthesize import cluster_claims

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )
    contested = [c for c in clusters if c.verdict == "CONTESTED"]

    assert len(contested) == 1, "the planted contradiction was not detected"
    cluster = contested[0]
    assert set(cluster.claim_ids) == {"S1-C01", "S3-C01"}
    assert set(cluster.agreeing_sources) ^ set(cluster.disagreeing_sources) == {"S1", "S3"}
    assert cluster.rationale.strip(), "a contested cluster must explain the disagreement"


def test_three_source_agreement_is_consensus(synthetic_claims):
    from researchbrief.synthesize import cluster_claims

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )
    match = [c for c in clusters if set(c.claim_ids) == {"S1-C02", "S2-C01", "S4-C01"}]

    assert len(match) == 1
    assert match[0].verdict == "CONSENSUS"
    assert set(match[0].agreeing_sources) == {"S1", "S2", "S4"}


def test_single_source_claim_is_outlier(synthetic_claims):
    from researchbrief.synthesize import cluster_claims

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )
    match = [c for c in clusters if c.claim_ids == ["S2-C02"]]

    assert len(match) == 1
    assert match[0].verdict == "OUTLIER"


def test_no_claim_is_lost(synthetic_claims):
    from researchbrief.synthesize import cluster_claims

    claims = _claims(synthetic_claims)
    clusters = cluster_claims(
        claims, _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )

    placed = [cid for c in clusters for cid in c.claim_ids]
    assert sorted(placed) == sorted(c.id for c in claims)
    assert len(placed) == len(set(placed)), "a claim was placed in two clusters"


def test_verdict_assignment_is_pure():
    """Verdicts come from code, not the model, so they must be deterministic."""
    from researchbrief.synthesize import assign_verdict

    a = assign_verdict(["S1-C01", "S3-C01"], [["S1-C01", "S3-C01"]],
                       {"S1-C01": "S1", "S3-C01": "S3"})
    b = assign_verdict(["S1-C01", "S3-C01"], [["S1-C01", "S3-C01"]],
                       {"S1-C01": "S1", "S3-C01": "S3"})
    assert a == b
    assert a[0] == "CONTESTED"


def test_two_claims_from_one_source_are_not_consensus():
    """A source agreeing with itself is not agreement between sources."""
    from researchbrief.synthesize import assign_verdict

    verdict, *_ = assign_verdict(
        ["S1-C01", "S1-C02"], [], {"S1-C01": "S1", "S1-C02": "S1"}
    )
    assert verdict == "OUTLIER"


def test_partial_coverage_gap_is_reported(synthetic_claims):
    from researchbrief.synthesize import cluster_claims, find_gaps

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )
    gaps = find_gaps(clusters, _sources(synthetic_claims))
    labels = {g.topic_label for g in gaps if g.kind == "partial_coverage"}

    assert "Where are AI productivity gains concentrated?" in labels


def test_failed_source_is_reported_as_unaddressed(synthetic_claims):
    """S5 was paywalled, so its position on every topic is unknown."""
    from researchbrief.synthesize import cluster_claims, find_gaps

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )
    gaps = find_gaps(clusters, _sources(synthetic_claims))

    assert any(g.kind == "unaddressed" and "S5" in g.silent_sources for g in gaps), (
        "a source that failed to load must be surfaced as a hole in the brief"
    )


def test_unusable_source_is_never_counted_as_silent_agreement(synthetic_claims):
    """Absence of a claim is not disagreement, and never agreement either."""
    from researchbrief.synthesize import cluster_claims

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([_grouping_response(synthetic_claims)]),
    )
    for cluster in clusters:
        assert "S5" not in cluster.agreeing_sources
        assert "S5" not in cluster.disagreeing_sources


def test_model_returning_unknown_claim_ids_does_not_corrupt_output(synthetic_claims):
    from researchbrief.synthesize import cluster_claims

    payload = json.loads(_grouping_response(synthetic_claims))
    payload["clusters"][0]["claim_ids"].append("S9-C99")

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims),
        "AI and jobs", FakeLLM([json.dumps(payload)]),
    )
    placed = {cid for c in clusters for cid in c.claim_ids}
    assert "S9-C99" not in placed, "a hallucinated claim id leaked into the output"


def test_orphan_claims_are_still_clustered(synthetic_claims):
    """If the model forgets a claim, it must land somewhere, not disappear."""
    from researchbrief.synthesize import cluster_claims

    payload = json.loads(_grouping_response(synthetic_claims))
    payload["clusters"] = payload["clusters"][:2]

    claims = _claims(synthetic_claims)
    clusters = cluster_claims(
        claims, _sources(synthetic_claims), "AI and jobs", FakeLLM([json.dumps(payload)]),
    )
    placed = {cid for c in clusters for cid in c.claim_ids}
    assert placed == {c.id for c in claims}


@pytest.mark.live
def test_real_model_detects_an_obvious_contradiction(synthetic_claims):
    from researchbrief.llm import LLM
    from researchbrief.synthesize import cluster_claims

    clusters = cluster_claims(
        _claims(synthetic_claims), _sources(synthetic_claims), "AI and jobs", LLM()
    )
    contested = [c for c in clusters if c.verdict == "CONTESTED"]
    assert contested, "the real model failed to group two directly opposing forecasts"

    ids = {cid for c in contested for cid in c.claim_ids}
    assert {"S1-C01", "S3-C01"} <= ids
