# Notes

## Approach

Five staged steps, not one prompt: fetch, extract, extract-claims, synthesize,
render. Each fails and retries differently (network vs schema vs reasoning), so
keeping them separate makes each testable. Only two stages call a model. Claim
extraction runs per source so it cannot smooth claims toward a false consensus;
synthesis sees every claim at once, labelled only by source id, so it groups by
content rather than by source authority. The model only groups and marks
opposition; the verdict (consensus / contested / outlier) is assigned in code, so
it is deterministic and testable. Every claim carries a verbatim quote and is
dropped if that quote is not found in the source, which is the anti-hallucination
guarantee.

## Assumptions (made rather than asked)

- Sources are given and trusted; the tool reports what each claims and who
  disagrees, it does not judge credibility.
- Absence of a claim is a coverage gap, not disagreement. These are reported
  separately so silence is never read as agreement.
- Contradiction is judged at the claim level (a magnitude difference counts).
- JS-only pages are detected and reported, not rendered.

## What I would do differently for production

- Cache the claim and synthesis stages like the fetch stage, so a rerun after a
  formatting change does not re-call the model.
- Add a headless-browser fallback for JS-only pages instead of reporting them as
  failures, and a small pool of extraction strategies per domain.
- Make synthesis more robust at scale: chunk the claim set and reconcile, rather
  than one call, once corpora exceed a few dozen sources.
- Observability: structured run metrics, retries and cost per source shipped to a
  dashboard, not just a run.log.

## Tradeoffs from the 2-hour budget

- No headless browser; JS shells are classified and surfaced in the ledger
  instead of rendered. This was judged a poor use of the budget relative to the
  reasoning layer, which is what is being evaluated.
- Single synthesis call over a bounded claim set, no vector store or clustering
  library.
- Free-tier Gemini flash for both stages (Pro has no free quota); the provider
  sits behind one client, so swapping to a stronger model is a one-line change.
- Contradiction detection depends on the model marking oppositions; the code
  guarantees structure and determinism, but recall of subtle disagreements is
  only as good as the grouping call.
