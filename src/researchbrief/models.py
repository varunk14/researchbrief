"""Data contracts shared across every stage.

These dataclasses are the frozen interface between stages. They carry no logic;
each stage reads and writes them and nothing else crosses a module boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FetchStatus(str, Enum):
    """Outcome of a single fetch, used to decide whether a source is usable."""

    OK = "OK"
    PAYWALLED = "PAYWALLED"
    BLOCKED = "BLOCKED"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    EMPTY = "EMPTY"
    ERROR = "ERROR"


@dataclass
class FetchResult:
    """Result of fetching one URL. Produced even when the fetch fails."""

    url: str
    final_url: str | None
    status: FetchStatus
    http_status: int | None
    html: str | None
    elapsed_ms: int
    fetched_at: str  # ISO 8601
    from_cache: bool = False
    error: str | None = None


@dataclass
class Document:
    """Cleaned text plus metadata for one source, with a usability verdict."""

    source_id: str
    url: str
    title: str | None
    site_name: str | None
    author: str | None
    published_at: str | None
    text: str
    word_count: int
    extractor: str  # "trafilatura" | "readability" | "raw_p" | "none"
    fetch_status: FetchStatus
    quality: float  # 0.0 - 1.0
    usable: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class Claim:
    """A single atomic, source-attributable assertion about the topic."""

    id: str  # "S1-C03"
    source_id: str  # "S1"
    text: str
    quote: str  # verbatim span from Document.text
    claim_type: str  # empirical | forecast | causal | normative | definitional
    confidence: float


@dataclass
class ClaimCluster:
    """A group of claims addressing one underlying question, with a verdict."""

    id: str  # "T01"
    topic_label: str  # phrased as a question
    claim_ids: list[str]
    verdict: str  # CONSENSUS | CONTESTED | OUTLIER
    agreeing_sources: list[str] = field(default_factory=list)
    disagreeing_sources: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class Gap:
    """A topic that the corpus covers unevenly or not at all."""

    topic_label: str
    kind: str  # partial_coverage | unaddressed
    covered_by: list[str] = field(default_factory=list)
    silent_sources: list[str] = field(default_factory=list)


@dataclass
class Brief:
    """The full deliverable: everything needed to render brief.md and brief.json."""

    topic: str
    generated_at: str
    documents: list[Document] = field(default_factory=list)
    fetch_results: list[FetchResult] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    clusters: list[ClaimCluster] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
