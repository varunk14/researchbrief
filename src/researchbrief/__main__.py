"""Command-line entrypoint and pipeline orchestration.

Wires the five stages together, writes brief.md, brief.json and run.log, and maps
outcomes to exit codes: 0 success, 1 unhandled error, 2 bad arguments, 3 a run
that completed with fewer than two usable sources (too few for cross-source
reasoning, which the tool reports loudly rather than faking).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from .claims import extract_all_claims
from .config import Config, read_url_file
from .extract import to_document
from .fetch import fetch_all
from .llm import LLM
from .models import Document
from .render import build_brief, to_json, to_markdown
from .synthesize import cluster_claims, find_gaps

MIN_USABLE_SOURCES = 2

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_BAD_ARGS = 2
EXIT_TOO_FEW_SOURCES = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchbrief",
        description=(
            "Build a cross-source research brief from a topic and 3-5 source URLs. "
            "The brief separates what sources agree on, disagree on, and leave uncovered."
        ),
    )
    parser.add_argument("--topic", required=True, help="the question or subject to brief")
    parser.add_argument(
        "--urls",
        required=True,
        action="append",
        metavar="FILE_OR_URL",
        help="a file of newline-delimited URLs, or a bare URL (repeatable)",
    )
    parser.add_argument("--out", default="out", help="output directory (default: out)")
    parser.add_argument(
        "--verbose", action="store_true", help="print a progress line per stage"
    )
    return parser


def _resolve_urls(entries: list[str]) -> list[str]:
    urls: list[str] = []
    for entry in entries:
        path = Path(entry)
        if path.exists():
            urls.extend(read_url_file(path))
        else:
            urls.append(entry.strip())
    return urls


def _domain(url: str) -> str:
    return urlparse(url).netloc or url


def _document_to_source(doc: Document) -> dict:
    return {
        "source_id": doc.source_id,
        "url": doc.url,
        "title": doc.title,
        "domain": _domain(doc.url),
        "status": doc.fetch_status.value,
        "extractor": doc.extractor,
        "quality": doc.quality,
        "usable": doc.usable,
        "published_at": doc.published_at,
        "notes": doc.notes,
    }


def run(args: argparse.Namespace) -> int:
    cfg = Config()
    log: list[str] = []
    started = time.monotonic()

    def note(message: str) -> None:
        log.append(message)
        if args.verbose:
            print(message, file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = _resolve_urls(args.urls)
    note(f"stage 1 fetch: {len(urls)} url(s)")
    t = time.monotonic()
    fetch_results = fetch_all(urls, cfg)
    for r in fetch_results:
        note(f"  {r.status.value:9} {r.url}")
    note(f"stage 1 fetch done in {time.monotonic() - t:.1f}s")

    documents = [
        to_document(fr, source_id=f"S{i + 1}", cfg=cfg)
        for i, fr in enumerate(fetch_results)
    ]
    sources = [_document_to_source(d) for d in documents]
    usable = [d for d in documents if d.usable]
    note(f"stage 2 extract: {len(usable)}/{len(documents)} usable")

    llm = LLM(cfg)
    if len(usable) >= MIN_USABLE_SOURCES:
        t = time.monotonic()
        claims = extract_all_claims(documents, args.topic, llm, cfg)
        note(f"stage 3 claims: {len(claims)} claim(s) in {time.monotonic() - t:.1f}s")
        t = time.monotonic()
        clusters = cluster_claims(claims, sources, args.topic, llm)
        note(f"stage 4 synthesize: {len(clusters)} cluster(s) in {time.monotonic() - t:.1f}s")
    else:
        claims = []
        clusters = []
        note("stage 3-4 skipped: fewer than two usable sources")

    gaps = find_gaps(clusters, sources)

    stats = {
        "provider": cfg.provider,
        "models": {"fast": cfg.fast_model, "smart": cfg.smart_model},
        "tokens": {
            "input": llm.usage.input_tokens,
            "output": llm.usage.output_tokens,
            "total": llm.usage.total_tokens,
            "calls": llm.usage.calls,
        },
        "counts": {
            "urls": len(urls),
            "usable_sources": len(usable),
            "claims": len(claims),
            "clusters": len(clusters),
        },
        "elapsed_s": round(time.monotonic() - started, 2),
    }

    brief = build_brief(args.topic, sources, claims, clusters, gaps, stats=stats)
    (out_dir / "brief.md").write_text(to_markdown(brief), encoding="utf-8")
    (out_dir / "brief.json").write_text(to_json(brief), encoding="utf-8")
    log.append(f"total tokens: {llm.usage.total_tokens} over {llm.usage.calls} call(s)")
    log.append(f"total elapsed: {stats['elapsed_s']}s")
    (out_dir / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")

    if len(usable) < MIN_USABLE_SOURCES:
        print(
            f"Only {len(usable)} of {len(documents)} sources are usable; at least "
            f"{MIN_USABLE_SOURCES} are required for cross-source reasoning. "
            f"See {out_dir / 'brief.md'} for the source ledger.",
            file=sys.stderr,
        )
        return EXIT_TOO_FEW_SOURCES

    print(f"Wrote {out_dir / 'brief.md'}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # argparse exits 2 on bad arguments
    try:
        return run(args)
    except Exception as exc:  # last-resort guard so the CLI never dumps a traceback
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
