# researchbrief

A command-line agent that reads a topic and 3-5 source URLs and produces a
**cross-source brief**: what the sources agree on, where they disagree, what only
one source says, and what nobody covers. The point is cross-source reasoning, not
a stack of per-source summaries.

## What it produces

Written to the output directory:

| File | Purpose |
|---|---|
| `brief.md` | The human-readable deliverable. |
| `brief.json` | The same data, machine-readable. |
| `run.log` | Per-stage timings, fetch outcomes, model and token counts. |

## Install

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `python -m researchbrief` runnable
```

## Configure

Copy the example environment file and add a key for your provider:

```bash
cp .env.example .env
```

`.env` selects the model provider and holds its key:

```
LLM_PROVIDER=gemini          # gemini (default) or anthropic
GEMINI_API_KEY=...           # for the default provider
ANTHROPIC_API_KEY=...        # only if LLM_PROVIDER=anthropic
```

Gemini is the default because it has a free tier; set `LLM_PROVIDER=anthropic`
and `ANTHROPIC_API_KEY` to use Anthropic instead. Keys are read from the
environment; load `.env` in your shell (for example with `set -a; . ./.env; set +a`).

## Run

```bash
python -m researchbrief --topic "How AI is reshaping entry-level jobs" \
                        --urls samples/links.txt \
                        --out out/
```

`--urls` takes a newline-delimited file (blank lines and `#` comments ignored) or
one or more bare URLs (the flag may be repeated). Add `--verbose` for a progress
line per stage. `--help` documents every flag.

Exit codes: `0` success, `1` unhandled error, `2` bad arguments, `3` completed
but fewer than two usable sources (too few to reason across).

## Sample output

```markdown
# Research brief: How AI is reshaping entry-level jobs

Generated: 2026-08-11T12:00:00Z
Sources: 4 of 5 sources usable

## What the sources agree on
### Where are AI productivity gains concentrated? [S1][S2][S4]
- AI productivity gains are concentrated among high-skill occupations. [S1]

## Where the sources disagree
### Will AI increase or decrease net employment?
- AI will increase net employment over the next decade. [S1]
- AI will decrease net employment over the next decade. [S3]
- Rationale: Sources disagree on the direction of net employment change.
```

## How it works

Five staged steps, each independently testable and each cached so a rerun
resumes without refetching or re-calling the model:

```
URLs -> [1 FETCH] -> [2 EXTRACT] -> [3 CLAIMS] -> [4 SYNTHESIZE] -> [5 RENDER] -> brief.md
         network      no model      1 call/src    1 call/corpus     no model
```

Staging matters because each step fails differently: a fetch failure is a network
problem, a claim failure is a schema problem, a synthesis failure is a reasoning
problem. Claim extraction runs per source so it cannot smooth claims toward a
false consensus; synthesis sees every claim at once so it can detect
contradiction. Every claim carries a verbatim quote from its source, and any
claim whose quote is not found in the text is dropped.

## Tests

```bash
pip install -r requirements.txt
pytest                 # offline, deterministic
pytest --live          # also runs network and real model calls
```
