"""Fetch layer.

Pulls bytes off the network and classifies the outcome without ever raising:
every input URL yields exactly one FetchResult, whatever the network does. A
disk cache lets a rerun resume without touching the network.
"""
from __future__ import annotations

import ipaddress
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .config import Config
from .models import FetchResult, FetchStatus

PAYWALL_MARKERS = ("subscribe to continue", "metered", "paywall", "sign in to read")
# Match genuine bot-wall interstitials, not any page that merely mentions
# "cloudflare" or "captcha" in a script or article body.
BLOCK_MARKERS = (
    "attention required! | cloudflare",
    "checking your browser before accessing",
    "cf-browser-verification",
    "please verify you are a human",
    "enable javascript and cookies to continue",
    "verifying you are human",
)
MIN_OK_BYTES = 500
MAX_REDIRECTS = 6


class _BlockedRedirect(Exception):
    """Raised when a redirect would leave http(s) or point at a local address."""


def classify(http_status: int, body: str | None) -> FetchStatus:
    """Map an HTTP status and body to a FetchStatus. Pure; no network.

    Returns ERROR for any status not otherwise recognised (for example 5xx).
    """
    text = (body or "").lower()
    if http_status in (404, 410):
        return FetchStatus.NOT_FOUND
    if http_status in (401, 402, 403):
        return FetchStatus.PAYWALLED
    if http_status == 429:
        return FetchStatus.BLOCKED
    if 200 <= http_status < 300:
        if any(marker in text for marker in PAYWALL_MARKERS):
            return FetchStatus.PAYWALLED
        if any(marker in text for marker in BLOCK_MARKERS):
            return FetchStatus.BLOCKED
        if len(body or "") < MIN_OK_BYTES:
            return FetchStatus.EMPTY
        return FetchStatus.OK
    return FetchStatus.ERROR


def _is_blocked_target(url: str) -> bool:
    """True if the URL is not http(s) or resolves to a local/private address."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.hostname or ""
    if host in ("localhost", "") or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a hostname; DNS is not our job to second-guess here
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def _cache_path(url: str, cfg: Config) -> Path:
    digest = sha256(url.encode("utf-8")).hexdigest()
    return Path(cfg.cache_dir) / "fetch" / f"{digest}.json"


def _read_cache(url: str, cfg: Config) -> FetchResult | None:
    path = _cache_path(url, cfg)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    data["status"] = FetchStatus(data["status"])
    data["from_cache"] = True
    return FetchResult(**data)


def _write_cache(url: str, cfg: Config, result: FetchResult) -> None:
    path = _cache_path(url, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.__dict__.copy()
    payload["status"] = result.status.value
    payload["from_cache"] = False
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # a cache write failure must not fail the fetch


def _http_get(url: str, cfg: Config) -> tuple[str, int, str]:
    """Fetch a URL, following redirects manually so each hop can be checked.

    Returns (final_url, status_code, body). Raises httpx errors on network
    failure and _BlockedRedirect if a hop leaves http(s) or targets a local host.
    """
    headers = {"User-Agent": cfg.user_agent}
    with httpx.Client(follow_redirects=False, timeout=cfg.timeout, headers=headers) as client:
        current = url
        response = None
        for _ in range(MAX_REDIRECTS):
            if _is_blocked_target(current):
                raise _BlockedRedirect(current)
            response = client.get(current)
            if response.is_redirect and response.next_request is not None:
                current = str(response.next_request.url)
                continue
            return str(response.url), response.status_code, response.text
        assert response is not None
        return str(response.url), response.status_code, response.text


def fetch_one(url: str, cfg: Config | None = None) -> FetchResult:
    """Fetch one URL, returning a FetchResult. Never raises.

    On cache hit the network is skipped and from_cache is True. 5xx responses,
    connection errors and read timeouts are retried with exponential backoff up
    to cfg.max_retries; 4xx responses are never retried.
    """
    cfg = cfg or Config()
    cached = _read_cache(url, cfg)
    if cached is not None:
        return cached

    start = time.monotonic()
    result = _fetch_with_retries(url, cfg, start)
    _write_cache(url, cfg, result)
    return result


def _build(
    url: str,
    start: float,
    status: FetchStatus,
    *,
    final_url: str | None = None,
    http_status: int | None = None,
    html: str | None = None,
    error: str | None = None,
) -> FetchResult:
    return FetchResult(
        url=url,
        final_url=final_url,
        status=status,
        http_status=http_status,
        html=html,
        elapsed_ms=int((time.monotonic() - start) * 1000),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        from_cache=False,
        error=error,
    )


def _fetch_with_retries(url: str, cfg: Config, start: float) -> FetchResult:
    for attempt in range(cfg.max_retries + 1):
        last = attempt == cfg.max_retries
        try:
            final_url, code, body = _http_get(url, cfg)
        except httpx.TimeoutException as exc:
            if last:
                return _build(url, start, FetchStatus.TIMEOUT, error=str(exc) or "timeout")
        except _BlockedRedirect as exc:
            return _build(url, start, FetchStatus.ERROR, error=f"blocked redirect to {exc}")
        except (httpx.HTTPError, OSError) as exc:
            if last:
                return _build(url, start, FetchStatus.ERROR, error=str(exc) or repr(exc))
        else:
            if code >= 500 and not last:
                time.sleep(_backoff(attempt))
                continue
            status = classify(code, body)
            return _build(
                url, start, status, final_url=final_url, http_status=code, html=body
            )
        time.sleep(_backoff(attempt))
    # Unreachable: the loop always returns on its final attempt.
    return _build(url, start, FetchStatus.ERROR, error="exhausted retries")


def _backoff(attempt: int) -> float:
    return 0.5 * (2 ** attempt)


def fetch_all(urls: list[str], cfg: Config | None = None) -> list[FetchResult]:
    """Fetch every URL in parallel (capped) and return results in input order.

    Never raises; a failure inside any single fetch becomes an ERROR result so
    the returned list always has one entry per input URL, in order.
    """
    cfg = cfg or Config()
    if not urls:
        return []

    def _safe(url: str) -> FetchResult:
        try:
            return fetch_one(url, cfg)
        except Exception as exc:  # defensive: fetch_one already never raises
            return _build(url, time.monotonic(), FetchStatus.ERROR, error=repr(exc))

    workers = max(1, min(cfg.fetch_concurrency, len(urls)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_safe, urls))
