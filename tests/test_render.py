"""Brief rendering.

Rendering is deterministic and makes no model call. These tests check that the
brief is honest: every citation resolves, every failed source stays visible,
and an empty contradiction section says so rather than being omitted.
"""
from __future__ import annotations

import re

import pytest

from helpers import FakeLLM

pytestmark = pytest.mark.render

SECTIONS = [
    "agree",
    "disagree",
    "single-source",
    "coverage",
    "source ledger",
]


@pytest.fixture
def brief(synthetic_claims):
    import json

    from researchbrief.models import Claim
    from researchbrief.synthesize import cluster_claims, find_gaps
    from researchbrief.render import build_brief

    claims = [Claim(**c) for c in synthetic_claims["claims"]]
    sources = synthetic_claims["sources"]
    response = json.dumps({"clusters": [
        {"topic_label": g["topic_label"], "claim_ids": g["claim_ids"],
         "opposing_pairs": g["opposing_pairs"], "rationale": "fixture rationale"}
        for g in synthetic_claims["expected_grouping"]
    ]})

    clusters = cluster_claims(claims, sources, "AI and jobs", FakeLLM([response]))
    return build_brief(
        topic="AI and jobs",
        sources=sources,
        claims=claims,
        clusters=clusters,
        gaps=find_gaps(clusters, sources),
    )


def test_all_sections_present_and_ordered(brief):
    from researchbrief.render import to_markdown

    text = to_markdown(brief).lower()
    positions = []
    for name in SECTIONS:
        idx = text.find(name)
        assert idx != -1, f"missing section: {name}"
        positions.append(idx)

    assert positions == sorted(positions), "sections are out of order"


def test_disagreement_section_precedes_outliers(brief):
    from researchbrief.render import to_markdown

    text = to_markdown(brief).lower()
    assert text.find("disagree") < text.find("single-source")


def test_every_citation_resolves_to_the_ledger(brief):
    from researchbrief.render import to_markdown

    text = to_markdown(brief)
    cited = set(re.findall(r"\[(S\d+)\]", text))
    known = {s["source_id"] for s in brief.sources} if isinstance(brief.sources[0], dict) \
        else {s.source_id for s in brief.sources}

    assert cited, "the brief contains no citations at all"
    assert cited <= known, f"orphan citations: {cited - known}"


def test_failed_source_appears_in_the_ledger_with_a_reason(brief):
    from researchbrief.render import to_markdown

    text = to_markdown(brief)
    assert "S5" in text, "a source that failed to load was dropped from the brief"
    assert "PAYWALLED" in text.upper()


def test_contested_cluster_shows_both_sides(brief):
    from researchbrief.render import to_markdown

    text = to_markdown(brief)
    start = text.lower().find("disagree")
    end = text.lower().find("single-source")
    section = text[start:end]

    assert "S1" in section and "S3" in section, "only one side of the disagreement rendered"


def test_empty_contested_section_is_explicit(brief):
    from researchbrief.render import to_markdown

    stripped = brief.__class__(**{**brief.__dict__, "clusters": [
        c for c in brief.clusters if c.verdict != "CONTESTED"
    ]})
    text = to_markdown(stripped).lower()

    assert "disagree" in text, "the section was omitted instead of reporting nothing found"
    assert "no " in text


def test_no_emoji_in_output(brief):
    from researchbrief.render import to_markdown

    emoji = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F]")
    assert not emoji.search(to_markdown(brief))


def test_header_reports_usable_source_count(brief):
    from researchbrief.render import to_markdown

    header = to_markdown(brief)[:600]
    assert "4" in header and "5" in header, "header must state usable of total sources"


def test_json_round_trips(brief):
    from researchbrief.render import to_json, from_json

    assert from_json(to_json(brief)) == brief


def test_rendering_is_deterministic(brief):
    from researchbrief.render import to_markdown

    first = to_markdown(brief)
    second = to_markdown(brief)
    assert first == second
