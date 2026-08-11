"""Claim extraction: one model call per usable document.

The central guarantee is quote verification. Every claim must carry a quote that
is a verbatim span of the source text; any claim whose quote cannot be found is
dropped, so a claim in the brief always traces back to something the source
actually said. Unusable documents are never sent to the model.
"""
from __future__ import annotations

import json
import re

from .config import Config
from .models import Claim, Document

MAX_CLAIMS = 20

CLAIMS_PROMPT = """You extract atomic, source-attributable claims about a topic.

Topic: {topic}

Read the SOURCE TEXT below and return claims as JSON of the form:
{{"claims": [{{"text": "...", "quote": "...", "claim_type": "...", "confidence": 0.0}}]}}

Rules for each claim:
- Atomic: one assertion only. Split "X and Y" into two claims.
- Self-contained: no leading pronoun (it, this, they, he, she, such) and no
  unnamed reference. A reader who has not seen the source must understand it.
- quote: a VERBATIM span copied from the source text that supports the claim.
  Copy it exactly, do not paraphrase.
- claim_type is one of: empirical, forecast, causal, normative, definitional.
- confidence is your faithfulness confidence from 0.0 to 1.0.
- Return between 5 and 15 claims. Do not summarize the whole document.

Return only JSON, no prose.

SOURCE TEXT:
{text}
"""

RETRY_SUFFIX = """

Your previous response could not be parsed as JSON.
Parser error: {error}
Return only a single valid JSON object of the required shape, nothing else.
"""


def extract_claims(document: Document, topic: str, llm) -> list[Claim]:
    """Extract quote-verified claims from one document.

    Returns an empty list for an unusable document (no model call), or when the
    model returns unparseable JSON twice. Never raises on model output.
    """
    if not document.usable:
        return []

    prompt = CLAIMS_PROMPT.format(topic=topic, text=document.text)
    raw = llm.call_json(prompt, tier="fast")
    try:
        payload = _parse(raw)
    except ValueError as exc:
        retry_prompt = prompt + RETRY_SUFFIX.format(error=str(exc))
        raw = llm.call_json(retry_prompt, tier="fast")
        try:
            payload = _parse(raw)
        except ValueError:
            return []

    return _build_claims(payload, document)


def _parse(raw: str) -> dict:
    """Parse model output into a dict, tolerating fences and surrounding prose.

    Raises ValueError with a short reason if no JSON object can be recovered.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON object: {exc.msg}") from exc

    raise ValueError("no JSON object found in response")


def _build_claims(payload: dict, document: Document) -> list[Claim]:
    raw_claims = payload.get("claims") if isinstance(payload, dict) else None
    if not isinstance(raw_claims, list):
        return []

    haystack = _normalise(document.text)
    claims: list[Claim] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", "")).strip()
        text = str(item.get("text", "")).strip()
        if not quote or not text:
            continue
        if _normalise(quote) not in haystack:
            continue  # anti-hallucination: unverifiable quote is dropped
        index = len(claims) + 1
        claims.append(
            Claim(
                id=f"{document.source_id}-C{index:02d}",
                source_id=document.source_id,
                text=text,
                quote=quote,
                claim_type=str(item.get("claim_type", "empirical")).strip() or "empirical",
                confidence=_as_float(item.get("confidence")),
            )
        )
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _as_float(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def extract_all_claims(
    documents: list[Document], topic: str, llm, cfg: Config | None = None
) -> list[Claim]:
    """Extract claims for every usable document. Orchestration helper for the CLI.

    Runs documents concurrently (capped) and returns their claims flattened, in
    source order. A document that yields nothing simply contributes no claims.
    """
    cfg = cfg or Config()
    from concurrent.futures import ThreadPoolExecutor

    usable = [d for d in documents if d.usable]
    if not usable:
        return []
    workers = max(1, min(cfg.claim_concurrency, len(usable)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda d: extract_claims(d, topic, llm), usable))

    order = {d.source_id: i for i, d in enumerate(documents)}
    flattened = [claim for group in results for claim in group]
    flattened.sort(key=lambda c: (order.get(c.source_id, 0), c.id))
    return flattened
