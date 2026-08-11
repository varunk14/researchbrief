"""Test helpers shared across suites."""
from __future__ import annotations

import json
import re
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def load_html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def load_json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    """Whitespace-normalise, for verbatim-substring checks."""
    return re.sub(r"\s+", " ", text).strip()


class FakeLLM:
    """Returns canned responses in order. Records the prompts it was given."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def call_json(self, prompt: str, **_kwargs) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM called more times than it has responses")
        return self._responses.pop(0)


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")
