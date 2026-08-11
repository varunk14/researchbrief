"""Shared test configuration.

Suites are selected by marker, so `pytest -m synthesize` runs only the
cross-source reasoning tests. Tests that hit the network or a real model are
marked `live` and are skipped unless `--live` is passed, which keeps the
default run offline and deterministic.
"""
from __future__ import annotations

import pytest

from helpers import load_html, load_json, load_text

MARKERS = [
    "fetch: fetch layer",
    "extract: HTML to clean text",
    "claims: per-source claim extraction",
    "synthesize: cross-source reasoning",
    "render: brief output",
    "cli: command line and end-to-end",
    "live: requires network or a real model call",
]


def pytest_configure(config):
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests that make real network and model calls",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_html() -> str:
    return load_html("article_good.html")


@pytest.fixture
def paywall_html() -> str:
    return load_html("paywall_stub.html")


@pytest.fixture
def js_shell_html() -> str:
    return load_html("js_shell.html")


@pytest.fixture
def malformed_html() -> str:
    return load_html("malformed.html")


@pytest.fixture
def recorded_claims_response() -> str:
    """A recorded model response, so claim extraction is deterministic."""
    return load_text("llm_claims_response.json")


@pytest.fixture
def synthetic_claims():
    """Claim set with planted structure. See fixtures/synthetic_claims.json."""
    return load_json("synthetic_claims.json")


