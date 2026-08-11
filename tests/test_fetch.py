"""Fetch layer.

The fetcher must classify real-world outcomes correctly and must never raise,
whatever the network does to it.
"""
from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.fetch

SAMPLE_URLS = "samples/links.txt"
BAD_HOST = "https://this-host-does-not-exist-9f3a.example/article"


def test_module_imports():
    from researchbrief import fetch, models  # noqa: F401


def test_fetch_result_contract():
    from researchbrief.models import FetchResult, FetchStatus

    required = {
        "url", "final_url", "status", "http_status",
        "html", "elapsed_ms", "fetched_at", "from_cache", "error",
    }
    assert required <= set(FetchResult.__dataclass_fields__)
    assert {
        "OK", "PAYWALLED", "BLOCKED", "NOT_FOUND", "TIMEOUT", "EMPTY", "ERROR",
    } <= {s.name for s in FetchStatus}


@pytest.mark.parametrize(
    "http_status,body,expected",
    [
        (200, "x" * 2000, "OK"),
        (402, "payment required", "PAYWALLED"),
        (403, "forbidden", "PAYWALLED"),
        (200, "Subscribe to continue reading. " * 5, "PAYWALLED"),
        (429, "too many requests", "BLOCKED"),
        (404, "not found", "NOT_FOUND"),
        (200, "tiny", "EMPTY"),
        (500, "server error", "ERROR"),
    ],
)
def test_status_classification(http_status, body, expected):
    from researchbrief.fetch import classify

    assert classify(http_status, body).name == expected


def test_classification_is_pure():
    """Classification must not depend on the network, so it can be tested."""
    from researchbrief.fetch import classify

    assert classify(200, "x" * 2000) == classify(200, "x" * 2000)


def test_unresolvable_host_returns_result_not_exception():
    from researchbrief.fetch import fetch_one
    from researchbrief.config import Config

    result = fetch_one(BAD_HOST, Config())
    assert result.status.name in {"ERROR", "TIMEOUT"}
    assert result.url == BAD_HOST
    assert result.error


def test_fetch_all_preserves_order_and_length():
    from researchbrief.fetch import fetch_all
    from researchbrief.config import Config

    urls = [BAD_HOST, BAD_HOST + "?2", BAD_HOST + "?3"]
    results = fetch_all(urls, Config())
    assert [r.url for r in results] == urls


def test_url_file_parsing_skips_comments_and_blanks(tmp_path):
    from researchbrief.config import read_url_file

    f = tmp_path / "urls.txt"
    f.write_text("# a comment\n\nhttps://a.example\n\n  https://b.example  \n# trailing\n")
    assert read_url_file(f) == ["https://a.example", "https://b.example"]


@pytest.mark.live
def test_sample_urls_all_return_a_result():
    from researchbrief.fetch import fetch_all
    from researchbrief.config import Config, read_url_file

    urls = read_url_file(SAMPLE_URLS)
    assert len(urls) == 5

    start = time.monotonic()
    results = fetch_all(urls, Config())
    elapsed = time.monotonic() - start

    assert len(results) == 5
    assert elapsed < 30, f"cold fetch took {elapsed:.1f}s, budget is 30s"
    assert sum(r.status.name == "OK" for r in results) >= 2, (
        "fewer than two sources fetched cleanly; the fetcher is too fragile"
    )


@pytest.mark.live
def test_second_run_is_served_from_cache():
    from researchbrief.fetch import fetch_all
    from researchbrief.config import Config, read_url_file

    urls = read_url_file(SAMPLE_URLS)
    cfg = Config()
    fetch_all(urls, cfg)

    start = time.monotonic()
    results = fetch_all(urls, cfg)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, f"cached run took {elapsed:.2f}s, expected under 1s"
    assert all(r.from_cache for r in results)
