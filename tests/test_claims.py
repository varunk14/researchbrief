"""Claim extraction.

The central guarantee here is quote verification: every claim must carry a
quote that is genuinely present in the source text. A claim that cannot be
traced back to the document is dropped rather than reported.
"""
from __future__ import annotations

import json

import pytest

from helpers import FIXTURES, FakeLLM, normalise

pytestmark = pytest.mark.claims


@pytest.fixture
def document(good_html):
    from researchbrief.extract import to_document
    from researchbrief.models import FetchResult, FetchStatus

    return to_document(
        FetchResult(
            url="https://example.org/a", final_url="https://example.org/a",
            status=FetchStatus.OK, http_status=200, html=good_html,
            elapsed_ms=1, fetched_at="2026-01-01T00:00:00Z",
        )
    )


def test_claim_contract():
    from researchbrief.models import Claim

    required = {"id", "source_id", "text", "quote", "claim_type", "confidence"}
    assert required <= set(Claim.__dataclass_fields__)


def test_recorded_response_parses_to_five_claims(document, recorded_claims_response):
    from researchbrief.claims import extract_claims

    claims = extract_claims(document, "AI and jobs", FakeLLM([recorded_claims_response]))
    assert len(claims) == 5


def test_every_quote_is_verbatim_in_the_source(document, recorded_claims_response):
    from researchbrief.claims import extract_claims

    haystack = normalise(document.text)
    for claim in extract_claims(document, "AI and jobs", FakeLLM([recorded_claims_response])):
        assert normalise(claim.quote) in haystack, (
            f"quote not found in source text: {claim.quote!r}"
        )


def test_fabricated_quotes_are_dropped_and_the_rest_survive(document):
    """One bad claim must not poison the other four."""
    from researchbrief.claims import extract_claims

    response = (FIXTURES / "llm_claims_hallucinated.json").read_text()
    claims = extract_claims(document, "AI and jobs", FakeLLM([response]))

    texts = " ".join(c.text for c in claims)
    assert "40 percent by 2031" not in texts, "fabricated claim was not dropped"
    assert len(claims) == 2, "surviving claims were lost along with the fabricated one"


def test_claim_ids_are_stable_and_readable(document, recorded_claims_response):
    from researchbrief.claims import extract_claims

    claims = extract_claims(document, "AI and jobs", FakeLLM([recorded_claims_response]))
    ids = [c.id for c in claims]
    assert len(set(ids)) == len(ids), "duplicate claim ids"
    for cid in ids:
        assert cid.startswith(document.source_id + "-C"), f"unreadable id: {cid}"


def test_claims_are_self_contained(document, recorded_claims_response):
    """A claim must stand alone; a leading pronoun means it does not."""
    from researchbrief.claims import extract_claims

    banned = ("it ", "this ", "they ", "these ", "he ", "she ", "such ")
    for claim in extract_claims(document, "AI and jobs", FakeLLM([recorded_claims_response])):
        first = claim.text.lower()
        assert not first.startswith(banned), f"claim is not self-contained: {claim.text!r}"


def test_malformed_json_triggers_retry_then_returns_empty(document):
    from researchbrief.claims import extract_claims

    bad = (FIXTURES / "llm_claims_malformed.txt").read_text()
    llm = FakeLLM([bad, bad])
    claims = extract_claims(document, "AI and jobs", llm)

    assert claims == []
    assert len(llm.calls) == 2, "expected exactly one retry after a parse failure"


def test_retry_prompt_includes_the_parse_error(document):
    from researchbrief.claims import extract_claims

    bad = (FIXTURES / "llm_claims_malformed.txt").read_text()
    llm = FakeLLM([bad, json.dumps({"claims": []})])
    extract_claims(document, "AI and jobs", llm)

    assert len(llm.calls) == 2
    assert llm.calls[1] != llm.calls[0], "retry sent an identical prompt"


def test_unusable_document_is_never_sent_to_the_model(js_shell_html):
    from researchbrief.claims import extract_claims
    from researchbrief.extract import to_document
    from researchbrief.models import FetchResult, FetchStatus

    doc = to_document(FetchResult(
        url="https://example.org/x", final_url="https://example.org/x",
        status=FetchStatus.OK, http_status=200, html=js_shell_html,
        elapsed_ms=1, fetched_at="2026-01-01T00:00:00Z",
    ))
    llm = FakeLLM([])
    assert extract_claims(doc, "AI and jobs", llm) == []
    assert llm.calls == [], "wasted a model call on an unusable document"


def test_claim_count_is_capped(document):
    from researchbrief.claims import extract_claims

    payload = {"claims": [{
        "text": f"Distinct assertion number {i} about the labour market.",
        "quote": "Senior workers in the same occupations saw employment hold steady or grow.",
        "claim_type": "empirical", "confidence": 0.8,
    } for i in range(60)]}

    claims = extract_claims(document, "AI and jobs", FakeLLM([json.dumps(payload)]))
    assert len(claims) <= 20


@pytest.mark.live
def test_real_model_extracts_verifiable_claims(document):
    from researchbrief.claims import extract_claims
    from researchbrief.llm import LLM

    claims = extract_claims(document, "AI and entry-level jobs", LLM())
    assert 3 <= len(claims) <= 20

    haystack = normalise(document.text)
    for claim in claims:
        assert normalise(claim.quote) in haystack
