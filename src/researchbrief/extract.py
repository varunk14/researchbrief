"""Extraction layer.

Turns fetched HTML into clean article text plus metadata, and scores how much
that text can be trusted. Junk (paywalls, JavaScript shells, link farms) is
identified and marked unusable rather than passed downstream as an article.

to_document never raises: malformed HTML, an empty string and None all return a
Document, so a failed source survives into the ledger instead of vanishing.
"""
from __future__ import annotations

import re

import trafilatura
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

from .config import Config
from .fetch import PAYWALL_MARKERS
from .models import Document, FetchResult, FetchStatus

_JS_SHELL_IDS = ("root", "app", "__next", "__nuxt")
_JS_SHELL_MAX_WORDS = 50
_LINK_DENSITY_LIMIT = 0.40
_MIN_PROSE_SENTENCES = 3
_SHORT_WORDS = 150


def to_document(
    fetch_result: FetchResult, source_id: str = "S1", cfg: Config | None = None
) -> Document:
    """Build a Document from a FetchResult. Never raises.

    On any extraction failure the returned Document has empty text, quality 0.0,
    extractor "none" and usable False; the fetch status is always preserved.
    """
    cfg = cfg or Config()
    html = fetch_result.html
    text, extractor, meta = _extract(html)

    notes: list[str] = []
    if text and len(text) > cfg.max_doc_chars:
        text, truncated = _truncate_at_sentence(text, cfg.max_doc_chars)
        if truncated:
            notes.append("truncated")

    word_count = len(text.split())
    quality = _score(html or "", text, word_count, notes)
    usable = quality >= 0.4

    return Document(
        source_id=source_id,
        url=fetch_result.url,
        title=meta.get("title"),
        site_name=meta.get("site_name"),
        author=meta.get("author"),
        published_at=meta.get("published_at"),
        text=text,
        word_count=word_count,
        extractor=extractor,
        fetch_status=fetch_result.status,
        quality=quality,
        usable=usable,
        notes=notes,
    )


def _extract(html: str | None) -> tuple[str, str, dict]:
    """Run the extractor cascade. Returns (text, extractor_name, metadata)."""
    if not html or not html.strip():
        return "", "none", {}

    for name, fn in (
        ("trafilatura", _try_trafilatura),
        ("readability", _try_readability),
        ("raw_p", _try_raw_paragraphs),
    ):
        try:
            text = fn(html)
        except Exception:
            text = ""
        if text and text.strip():
            return text.strip(), name, _metadata(html)

    return "", "none", _metadata(html)


def _try_trafilatura(html: str) -> str:
    return trafilatura.extract(
        html, include_comments=False, include_tables=False, favor_precision=True
    ) or ""


def _try_readability(html: str) -> str:
    summary_html = ReadabilityDocument(html).summary(html_partial=True)
    return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)


def _try_raw_paragraphs(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return "\n\n".join(p for p in paragraphs if p)


def _metadata(html: str) -> dict:
    """Best-effort metadata. Missing is fine; a wrong guess is not, so prefer None."""
    meta: dict = {"title": None, "site_name": None, "author": None, "published_at": None}
    try:
        extracted = trafilatura.extract_metadata(html)
    except Exception:
        extracted = None
    if extracted is not None:
        meta["title"] = _clean(getattr(extracted, "title", None))
        meta["site_name"] = _clean(getattr(extracted, "sitename", None))
        meta["author"] = _clean(getattr(extracted, "author", None))
        meta["published_at"] = _clean(getattr(extracted, "date", None))

    if not meta["title"]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                meta["title"] = _clean(soup.title.string)
            elif soup.h1:
                meta["title"] = _clean(soup.h1.get_text(" ", strip=True))
        except Exception:
            pass
    return meta


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    stripped = str(value).strip()
    return stripped or None


def _truncate_at_sentence(text: str, limit: int) -> tuple[str, bool]:
    """Cut text to the last sentence boundary at or before limit. Never mid-word."""
    if len(text) <= limit:
        return text, False
    cut = text.rfind(".", 0, limit)
    if cut == -1:
        cut = text.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit
        return text[:cut].rstrip(), True
    return text[: cut + 1], True


def _score(html: str, text: str, word_count: int, notes: list[str]) -> float:
    """Compute a 0.0-1.0 quality score and append explanatory notes in place."""
    if _is_js_shell(html, word_count):
        notes.append("js_shell")
        return 0.0
    if word_count == 0:
        return 0.0

    score = 1.0
    if word_count < _SHORT_WORDS:
        score = min(score, 0.3)
        notes.append("too_short")
    if _link_density(html) > _LINK_DENSITY_LIMIT:
        score -= 0.3
        notes.append("link_heavy")
    if _has_paywall_markers(html):
        score = min(score, 0.35)
        notes.append("likely_paywall")
    if _sentence_count(text) < _MIN_PROSE_SENTENCES:
        score -= 0.2
        notes.append("not_prose")

    return max(0.0, min(1.0, score))


def _is_js_shell(html: str, word_count: int) -> bool:
    if word_count >= _JS_SHELL_MAX_WORDS:
        return False
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return False
    for element_id in _JS_SHELL_IDS:
        if soup.find(attrs={"id": element_id}) is not None:
            return True
    return False


def _link_density(html: str) -> float:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return 0.0
    total = len(soup.get_text())
    if total == 0:
        return 0.0
    link_chars = sum(len(a.get_text()) for a in soup.find_all("a"))
    return link_chars / total


def _has_paywall_markers(html: str) -> bool:
    low = html.lower()
    return any(marker in low for marker in PAYWALL_MARKERS)


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.]", text))
