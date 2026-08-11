"""Brief rendering: deterministic formatting, no model call.

Turns the synthesised data into brief.md and brief.json. The brief is honest by
construction: every citation resolves to a ledger entry, a failed source stays
visible with its reason, and the disagreement section renders even when empty so
its absence is never mistaken for "not implemented".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from .models import Claim, ClaimCluster, Gap

COVERED = "x"
SILENT = "-"
NOT_APPLICABLE = "n/a"


@dataclass
class Brief:
    """Everything render needs. Sources are ledger dicts, one per input URL."""

    topic: str
    generated_at: str
    sources: list[dict]
    claims: list[Claim]
    clusters: list[ClaimCluster]
    gaps: list[Gap]
    stats: dict = field(default_factory=dict)


def build_brief(
    topic: str,
    sources: list[dict],
    claims: list[Claim],
    clusters: list[ClaimCluster],
    gaps: list[Gap],
    generated_at: str | None = None,
    stats: dict | None = None,
) -> Brief:
    """Assemble a Brief from the pipeline outputs."""
    return Brief(
        topic=topic,
        generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
        sources=list(sources),
        claims=list(claims),
        clusters=list(clusters),
        gaps=list(gaps),
        stats=stats or {},
    )


def _sget(source, key, default=None):
    """Read a field from a source that may be a dict or a dataclass."""
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _source_ids(brief: Brief) -> list[str]:
    return [_sget(s, "source_id") for s in brief.sources]


def _usable_ids(brief: Brief) -> set[str]:
    return {_sget(s, "source_id") for s in brief.sources if _sget(s, "usable")}


def _cite(source_ids) -> str:
    return "".join(f"[{sid}]" for sid in source_ids)


def _claims_by_id(brief: Brief) -> dict[str, Claim]:
    return {c.id: c for c in brief.claims}


def to_markdown(brief: Brief) -> str:
    """Render the six-section brief. Deterministic: same input, same output."""
    usable = _usable_ids(brief)
    parts: list[str] = []
    parts.append(f"# Research brief: {brief.topic}")
    parts.append("")
    parts.append(f"Generated: {brief.generated_at}")
    parts.append(f"Sources: {len(usable)} of {len(brief.sources)} sources usable")
    parts.append("")
    parts.append(_agree_section(brief))
    parts.append(_disagree_section(brief))
    parts.append(_outlier_section(brief))
    parts.append(_coverage_section(brief))
    parts.append(_ledger_section(brief))
    return "\n".join(parts).rstrip() + "\n"


def _agree_section(brief: Brief) -> str:
    consensus = [c for c in brief.clusters if c.verdict == "CONSENSUS"]
    consensus.sort(key=lambda c: len(c.agreeing_sources), reverse=True)
    lines = ["## What the sources agree on", ""]
    if not consensus:
        lines.append("No cross-source consensus was found among these sources.")
        return "\n".join(lines) + "\n"
    by_id = _claims_by_id(brief)
    for cluster in consensus:
        lines.append(f"### {cluster.topic_label} {_cite(cluster.agreeing_sources)}")
        for cid in cluster.claim_ids:
            claim = by_id.get(cid)
            if claim:
                lines.append(f"- {claim.text} [{claim.source_id}]")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _disagree_section(brief: Brief) -> str:
    contested = [c for c in brief.clusters if c.verdict == "CONTESTED"]
    lines = ["## Where the sources disagree", ""]
    if not contested:
        lines.append("No contradictions were found across these sources.")
        return "\n".join(lines) + "\n"
    by_id = _claims_by_id(brief)
    for cluster in contested:
        lines.append(f"### {cluster.topic_label}")
        for cid in cluster.claim_ids:
            claim = by_id.get(cid)
            if claim:
                lines.append(f"- {claim.text} [{claim.source_id}]")
        agree = _cite(cluster.agreeing_sources)
        disagree = _cite(cluster.disagreeing_sources)
        lines.append(f"- Positions: {agree} against {disagree}")
        if cluster.rationale.strip():
            lines.append(f"- Rationale: {cluster.rationale}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _outlier_section(brief: Brief) -> str:
    outliers = [c for c in brief.clusters if c.verdict == "OUTLIER"]
    lines = ["## Single-source claims", ""]
    if not outliers:
        lines.append("No single-source claims were recorded.")
        return "\n".join(lines) + "\n"
    lines.append("Asserted by one source only, unverified by others in this set.")
    lines.append("")
    by_id = _claims_by_id(brief)
    for cluster in outliers:
        for cid in cluster.claim_ids:
            claim = by_id.get(cid)
            if claim:
                lines.append(f"- {claim.text} [{claim.source_id}]")
    return "\n".join(lines).rstrip() + "\n"


def _coverage_section(brief: Brief) -> str:
    source_ids = _source_ids(brief)
    usable = _usable_ids(brief)
    lines = ["## Coverage and gaps", ""]

    header = "| Topic | " + " | ".join(source_ids) + " |"
    divider = "| --- | " + " | ".join("---" for _ in source_ids) + " |"
    lines.append(header)
    lines.append(divider)
    for cluster in brief.clusters:
        covering = {cid.split("-", 1)[0] for cid in cluster.claim_ids}
        cells = []
        for sid in source_ids:
            if sid not in usable:
                cells.append(NOT_APPLICABLE)
            elif sid in covering:
                cells.append(COVERED)
            else:
                cells.append(SILENT)
        label = cluster.topic_label or cluster.id
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")

    if brief.gaps:
        lines.append("Gaps:")
        for gap in brief.gaps:
            covered = ", ".join(gap.covered_by) or "none"
            silent = ", ".join(gap.silent_sources) or "none"
            lines.append(
                f"- {gap.kind}: {gap.topic_label} "
                f"(covered by {covered}; silent {silent})"
            )
    else:
        lines.append("No coverage gaps were detected.")
    return "\n".join(lines).rstrip() + "\n"


def _ledger_section(brief: Brief) -> str:
    counts: dict[str, int] = {}
    for claim in brief.claims:
        counts[claim.source_id] = counts.get(claim.source_id, 0) + 1

    lines = ["## Source ledger", ""]
    lines.append("| ID | Title | Domain | Published | Status | Extractor | Quality | Claims |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for source in brief.sources:
        sid = _sget(source, "source_id")
        title = _sget(source, "title") or "(no title)"
        domain = _sget(source, "domain") or _sget(source, "site_name") or "-"
        published = _sget(source, "published_at") or "n/a"
        status = _sget(source, "status") or "-"
        extractor = _sget(source, "extractor") or "-"
        quality = _sget(source, "quality")
        quality_str = f"{quality:.2f}" if isinstance(quality, (int, float)) else "-"
        claim_count = counts.get(sid, 0)
        lines.append(
            f"| {sid} | {title} | {domain} | {published} | {status} | "
            f"{extractor} | {quality_str} | {claim_count} |"
        )
    return "\n".join(lines) + "\n"


def to_json(brief: Brief) -> str:
    """Serialise a Brief to a JSON string that from_json reloads into an equal Brief."""
    payload = {
        "topic": brief.topic,
        "generated_at": brief.generated_at,
        "sources": brief.sources,
        "claims": [asdict(c) for c in brief.claims],
        "clusters": [asdict(c) for c in brief.clusters],
        "gaps": [asdict(g) for g in brief.gaps],
        "stats": brief.stats,
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)


def from_json(text: str) -> Brief:
    """Reload a Brief from to_json output."""
    data = json.loads(text)
    return Brief(
        topic=data["topic"],
        generated_at=data["generated_at"],
        sources=data["sources"],
        claims=[Claim(**c) for c in data["claims"]],
        clusters=[ClaimCluster(**c) for c in data["clusters"]],
        gaps=[Gap(**g) for g in data["gaps"]],
        stats=data.get("stats", {}),
    )
