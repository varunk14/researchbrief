"""Extraction layer.

Messy HTML must become clean text, and junk must be identified as junk rather
than passed downstream as though it were a full article.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.extract


def _doc(html: str, status: str = "OK"):
    from researchbrief.extract import to_document
    from researchbrief.models import FetchResult, FetchStatus

    return to_document(
        FetchResult(
            url="https://example.org/a",
            final_url="https://example.org/a",
            status=FetchStatus[status],
            http_status=200,
            html=html,
            elapsed_ms=1,
            fetched_at="2026-01-01T00:00:00Z",
        )
    )


def test_document_contract():
    from researchbrief.models import Document

    required = {
        "source_id", "url", "title", "site_name", "author", "published_at",
        "text", "word_count", "extractor", "fetch_status", "quality",
        "usable", "notes",
    }
    assert required <= set(Document.__dataclass_fields__)


def test_good_article_extracts_cleanly(good_html):
    doc = _doc(good_html)
    assert doc.usable is True
    assert doc.quality >= 0.7
    assert doc.word_count >= 200
    assert doc.title
    assert doc.extractor != "none"


def test_good_article_strips_boilerplate(good_html):
    doc = _doc(good_html)
    lowered = doc.text.lower()
    for chrome in ("privacy", "terms", "home"):
        assert chrome not in lowered, f"navigation or footer text survived: {chrome}"


def test_good_article_keeps_body_text(good_html):
    doc = _doc(good_html)
    assert "13 percent" in doc.text
    assert "codified knowledge" in doc.text


def test_paywall_stub_is_unusable(paywall_html):
    doc = _doc(paywall_html)
    assert doc.usable is False
    assert "likely_paywall" in doc.notes
    assert doc.quality < 0.4


def test_js_shell_is_detected(js_shell_html):
    doc = _doc(js_shell_html)
    assert doc.quality == 0.0
    assert "js_shell" in doc.notes
    assert doc.usable is False


def test_malformed_html_does_not_raise(malformed_html):
    doc = _doc(malformed_html)
    assert doc is not None
    assert isinstance(doc.text, str)


@pytest.mark.parametrize("html", ["", None, "   ", "<html></html>"])
def test_degenerate_input_returns_a_document(html):
    doc = _doc(html)
    assert doc.quality == 0.0
    assert doc.usable is False
    assert doc.extractor == "none"


def test_failed_fetch_still_produces_a_document():
    """A failed source must survive into the ledger, not vanish."""
    doc = _doc(None, status="TIMEOUT")
    assert doc.fetch_status.name == "TIMEOUT"
    assert doc.usable is False


def test_long_document_truncates_at_a_sentence_boundary():
    from researchbrief.config import Config

    body = "This is a sentence about artificial intelligence and work. " * 3000
    doc = _doc(f"<html><body><article><p>{body}</p></article></body></html>")
    if "truncated" in doc.notes:
        assert len(doc.text) <= Config().max_doc_chars
        assert doc.text.rstrip().endswith(".")


def test_quality_score_is_bounded(good_html, paywall_html, js_shell_html):
    for html in (good_html, paywall_html, js_shell_html):
        assert 0.0 <= _doc(html).quality <= 1.0
