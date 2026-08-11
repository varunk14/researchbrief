"""Cross-source synthesis: the reasoning layer.

One model call groups the whole claim set by the underlying question each claim
addresses. The model only groups and marks opposition; the verdict on each group
(consensus, contested, outlier) is assigned deterministically in code, so it is
testable and identical across runs. Every claim ends up in exactly one cluster.
"""
from __future__ import annotations

import json
import re

from .models import Claim, ClaimCluster, Gap

SYNTH_PROMPT = """You are grouping claims drawn from several sources about a topic.

Topic: {topic}

Each claim is labelled with only its source id, so group by CONTENT, not by which
source is more authoritative. Group claims that address the SAME underlying
question, even when worded differently. For every group give a short topic_label
phrased as a question, and list any opposing pairs of claim ids that directly
contradict each other.

Return JSON of the form:
{{"clusters": [{{"topic_label": "...?", "claim_ids": ["S1-C01", "S3-C01"],
  "opposing_pairs": [["S1-C01", "S3-C01"]], "rationale": "what the disagreement is about"}}]}}

Put every claim in exactly one group. Do not invent claim ids.

CLAIMS:
{claims}
"""


def cluster_claims(claims: list[Claim], sources, topic: str, llm) -> list[ClaimCluster]:
    """Group claims into clusters and assign each a verdict in code.

    Every input claim lands in exactly one cluster: claims the model omits or
    lists under an unknown id are placed in their own singleton cluster, and
    hallucinated ids are ignored. Order of clusters is model groups first, then
    orphans in input order.
    """
    id_to_source = {c.id: c.source_id for c in claims}
    all_ids = [c.id for c in claims]

    raw = llm.call_json(_build_prompt(claims, topic), tier="smart")
    groups = _parse_groups(raw)

    clusters: list[ClaimCluster] = []
    seen: set[str] = set()
    counter = 1

    for group in groups:
        ids = [
            cid
            for cid in group.get("claim_ids", [])
            if cid in id_to_source and cid not in seen
        ]
        if not ids:
            continue
        seen.update(ids)
        clusters.append(
            _make_cluster(
                counter,
                group.get("topic_label", ""),
                ids,
                group.get("opposing_pairs", []),
                group.get("rationale", ""),
                id_to_source,
                topic,
            )
        )
        counter += 1

    for claim in claims:
        if claim.id in seen:
            continue
        seen.add(claim.id)
        clusters.append(
            _make_cluster(
                counter, _as_question(claim.text), [claim.id], [], "", id_to_source, topic
            )
        )
        counter += 1

    placed = [cid for c in clusters for cid in c.claim_ids]
    assert sorted(placed) == sorted(all_ids), "a claim was lost during clustering"
    assert len(placed) == len(set(placed)), "a claim was placed in two clusters"
    return clusters


def _make_cluster(
    index: int,
    topic_label: str,
    claim_ids: list[str],
    opposing_pairs,
    rationale: str,
    id_to_source: dict[str, str],
    topic: str,
) -> ClaimCluster:
    verdict, agreeing, disagreeing = assign_verdict(claim_ids, opposing_pairs, id_to_source)
    if verdict == "CONTESTED" and not rationale.strip():
        rationale = f"Sources disagree on: {topic_label or topic}"
    return ClaimCluster(
        id=f"T{index:02d}",
        topic_label=topic_label,
        claim_ids=claim_ids,
        verdict=verdict,
        agreeing_sources=agreeing,
        disagreeing_sources=disagreeing,
        rationale=rationale if verdict == "CONTESTED" else "",
    )


def assign_verdict(
    claim_ids: list[str], opposing_pairs, id_to_source: dict[str, str]
) -> tuple[str, list[str], list[str]]:
    """Decide a cluster's verdict purely from its membership. No model, no state.

    Returns (verdict, agreeing_sources, disagreeing_sources). CONTESTED when the
    cluster contains an opposing pair from two different sources; otherwise
    CONSENSUS when two or more distinct sources agree; otherwise OUTLIER.
    """
    cid_set = set(claim_ids)
    distinct = sorted({id_to_source[c] for c in claim_ids if c in id_to_source})

    live_pairs = [
        pair
        for pair in (opposing_pairs or [])
        if len(pair) >= 2
        and pair[0] in cid_set
        and pair[1] in cid_set
        and id_to_source.get(pair[0]) != id_to_source.get(pair[1])
    ]

    if live_pairs:
        left = {id_to_source[p[0]] for p in live_pairs if p[0] in id_to_source}
        right = {id_to_source[p[1]] for p in live_pairs if p[1] in id_to_source}
        agreeing = sorted(left - right)
        disagreeing = sorted(right - left)
        return "CONTESTED", agreeing, disagreeing

    if len(distinct) >= 2:
        return "CONSENSUS", distinct, []
    return "OUTLIER", distinct, []


def find_gaps(clusters: list[ClaimCluster], sources) -> list[Gap]:
    """Report coverage holes, in code with no model call.

    partial_coverage: a topic two or more sources address but at least one usable
    source does not. unaddressed: sources that failed to load, whose position on
    every topic is therefore unknown and must not be read as silent agreement.
    """
    usable = {s["source_id"] for s in sources if s.get("usable")}
    failed = sorted(s["source_id"] for s in sources if not s.get("usable"))

    gaps: list[Gap] = []
    for cluster in clusters:
        covering = {cid.split("-", 1)[0] for cid in cluster.claim_ids}
        if len(covering) < 2:
            continue
        silent = sorted(usable - covering)
        if silent:
            gaps.append(
                Gap(
                    topic_label=cluster.topic_label,
                    kind="partial_coverage",
                    covered_by=sorted(covering),
                    silent_sources=silent,
                )
            )

    if failed:
        gaps.append(
            Gap(
                topic_label="Whole-topic coverage",
                kind="unaddressed",
                covered_by=sorted(usable),
                silent_sources=failed,
            )
        )
    return gaps


def _build_prompt(claims: list[Claim], topic: str) -> str:
    lines = [f"{c.id} ({c.source_id}): {c.text}" for c in claims]
    return SYNTH_PROMPT.format(topic=topic, claims="\n".join(lines))


def _parse_groups(raw: str) -> list[dict]:
    """Recover the clusters list from model output; on failure return []."""
    if not raw or not raw.strip():
        return []
    payload = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
    if not isinstance(payload, dict):
        return []
    groups = payload.get("clusters")
    return groups if isinstance(groups, list) else []


def _as_question(text: str) -> str:
    stripped = re.sub(r"\s+", " ", text).strip().rstrip(".")
    if not stripped:
        return "Ungrouped claim?"
    return stripped if stripped.endswith("?") else stripped + "?"
