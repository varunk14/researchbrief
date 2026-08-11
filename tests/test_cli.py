"""Command line interface and end-to-end behaviour.

Someone who has never seen this project must be able to clone it, follow the
README, and get a brief out of it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.cli

ROOT = Path(__file__).resolve().parents[1]


def _run(args, **kw):
    return subprocess.run(
        [sys.executable, "-m", "researchbrief", *args],
        capture_output=True, text=True, cwd=ROOT, **kw,
    )


def test_readme_exists_and_is_runnable():
    readme = (ROOT / "README.md")
    assert readme.exists(), "no README, so nobody can run this"

    text = readme.read_text().lower()
    for expected in ("install", "requirements.txt", "--topic", "anthropic_api_key"):
        assert expected in text, f"README does not mention {expected}"


def test_env_example_exists_and_has_no_real_key():
    example = ROOT / ".env.example"
    assert example.exists()
    assert example.read_text().strip().endswith("=")


def test_gitignore_covers_secrets_and_output():
    text = (ROOT / ".gitignore").read_text()
    for entry in (".env", "out/", "__pycache__/"):
        assert entry in text


def test_help_documents_every_flag():
    result = _run(["--help"])
    assert result.returncode == 0
    for flag in ("--topic", "--urls", "--out", "--verbose"):
        assert flag in result.stdout


def test_bad_arguments_exit_two():
    assert _run([]).returncode == 2


def test_too_few_usable_sources_exits_three(tmp_path):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://this-host-does-not-exist-9f3a.example\n")

    result = _run(["--topic", "anything", "--urls", str(urls), "--out", str(tmp_path / "out")])
    assert result.returncode == 3
    combined = (result.stdout + result.stderr).lower()
    assert "source" in combined, "the failure message does not explain what went wrong"


@pytest.mark.live
def test_end_to_end_produces_a_brief(tmp_path):
    assert os.environ.get("ANTHROPIC_API_KEY"), "set ANTHROPIC_API_KEY to run the live tests"

    out = tmp_path / "out"
    start = time.monotonic()
    result = _run([
        "--topic", "How AI is reshaping entry-level jobs",
        "--urls", "samples/links.txt",
        "--out", str(out),
    ])
    elapsed = time.monotonic() - start

    assert result.returncode == 0, result.stderr
    assert elapsed < 180, f"end-to-end run took {elapsed:.0f}s, budget is 180s"

    brief = (out / "brief.md").read_text()
    assert len(brief) > 1000
    assert (out / "brief.json").exists()
    assert (out / "run.log").exists()

    lowered = brief.lower()
    assert "agree" in lowered and "disagree" in lowered
