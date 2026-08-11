# Notes

Scope decisions and assumptions behind researchbrief.

## Assumptions

1. **Sources are given and trusted as inputs.** The agent does not judge source
   credibility. It reports what each source claims and who disagrees, not who is
   right.
2. **Contradiction is detected at the claim level, not the sentence level.** Two
   sources giving different numbers for the same quantity is a contradiction of
   magnitude and is reported as contested, with the kind of disagreement named.
3. **Absence of a claim is not disagreement.** A source that does not address a
   topic is a coverage gap, never an opposing view. Conflating the two is the
   most common failure of a naive implementation, so gaps are computed and
   reported separately.
4. **JS-only pages are detected and reported, not rendered.** A headless browser
   was judged a poor use of the build budget relative to the reasoning layer.
5. **English-language sources are assumed.**

## Non-goals

- No web search or source discovery; the URLs are provided.
- No headless browser; JavaScript-only shells are reported as failures.
- No vector database; clustering is a single model call over a bounded claim set.
- No web UI; Markdown is the deliverable.
- No multi-turn conversation.

## Design choices

- **Staged, not one big prompt.** Each stage fails differently and retries
  differently: a fetch failure is a network problem, a claim failure is a schema
  problem, a synthesis failure is a reasoning problem. One giant prompt makes all
  three indistinguishable and untestable.
- **Per-source extraction, whole-corpus synthesis.** Extraction never sees other
  sources, so it cannot smooth claims toward a consensus that was not in the
  text. Synthesis sees everything at once, so it can detect contradiction.
- **Verdict in code, grouping in the model.** The model groups claims and marks
  opposition; consensus/contested/outlier is decided deterministically in code,
  so the result is testable and identical across runs.
- **Quote verification.** Every claim carries a verbatim quote; a claim whose
  quote is not found in the source text is dropped. This is the anti-hallucination
  guarantee and it is cheap.

## Provider

The default model provider is Google Gemini (free tier); Anthropic is supported
as a fallback, selected by the `LLM_PROVIDER` environment variable. All model
access sits behind one small client, so no stage names a provider.

## What I would improve with more time

- Cache the claim and synthesis stages to disk like the fetch stage, so a rerun
  after a rendering tweak does not re-call the model.
- Support repeated bare `--urls` more richly (per-URL labels), and add a
  `--refresh` flag to bypass the fetch cache.
- Group forecasts by timeframe so contested forecasts distinguish "disagree on
  direction" from "disagree on when".
