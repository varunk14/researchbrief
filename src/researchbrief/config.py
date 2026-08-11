"""Runtime configuration and the URL-file parser.

Model ids are constants here, keyed by provider, so the provider can be swapped
in one place without touching any stage. Nothing in this module makes a network
or model call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Model ids per provider and tier. Stage 3 uses the fast tier, stage 4 the smart
# tier. Verify these against the provider's current docs before relying on them.
MODELS: dict[str, dict[str, str]] = {
    "gemini": {"fast": "gemini-2.5-flash", "smart": "gemini-2.5-pro"},
    "anthropic": {"fast": "claude-haiku-4-5-20251001", "smart": "claude-sonnet-5"},
}

DEFAULT_PROVIDER = "gemini"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class Config:
    """All tunables for a run. Instantiable with no arguments for tests."""

    user_agent: str = BROWSER_USER_AGENT
    timeout: float = 15.0
    max_retries: int = 2
    fetch_concurrency: int = 5
    claim_concurrency: int = 3
    cache_dir: str = "out/.cache"
    max_doc_chars: int = 40000
    provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER))

    @property
    def fast_model(self) -> str:
        """Model id for the cheap, per-source claim-extraction stage."""
        return MODELS[self.provider]["fast"]

    @property
    def smart_model(self) -> str:
        """Model id for the whole-corpus synthesis stage."""
        return MODELS[self.provider]["smart"]


def read_url_file(path: str | Path) -> list[str]:
    """Read a newline-delimited URL file.

    Blank lines and lines whose first non-space character is '#' are skipped.
    Surrounding whitespace on each URL is stripped. Returns the URLs in order.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    urls: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    return urls
