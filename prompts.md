# Prompts

Only two of the five pipeline stages call a model, so only two have prompts.
Fetch (stage 1), extraction (stage 2) and rendering (stage 5) are deterministic
code with no model call, on purpose: keeping those checks model-free is what
makes them testable and reproducible. The two model-facing stages below both run
at temperature 0 with JSON output and one retry that feeds the parse error back.

## Stage 3 - Claim extraction (fast model, one call per source)

Extracts atomic, self-contained, quote-backed claims from a single document.
The model never sees other sources, so it cannot smooth claims toward a
consensus that was not in the text. Every quote is verified as a verbatim span
of the source after the call; unverifiable claims are dropped in code.

```
You extract atomic, source-attributable claims about a topic.

Topic: {topic}

Read the SOURCE TEXT below and return claims as JSON of the form:
{"claims": [{"text": "...", "quote": "...", "claim_type": "...", "confidence": 0.0}]}

Rules for each claim:
- Atomic: one assertion only. Split "X and Y" into two claims.
- Self-contained: no leading pronoun (it, this, they, he, she, such) and no
  unnamed reference. A reader who has not seen the source must understand it.
- quote: a VERBATIM span copied from the source text that supports the claim.
  Copy it exactly, do not paraphrase.
- claim_type is one of: empirical, forecast, causal, normative, definitional.
- confidence is your faithfulness confidence from 0.0 to 1.0.
- Return between 5 and 15 claims. Do not summarize the whole document.

Return only JSON, no prose.

SOURCE TEXT:
{text}
```

Retry appends:

```
Your previous response could not be parsed as JSON.
Parser error: {error}
Return only a single valid JSON object of the required shape, nothing else.
```

## Stage 4 - Cross-source synthesis (smart model, one call over all claims)

Sees every claim at once, labelled only by source id so it cannot resolve a
disagreement by source authority. It groups claims by the underlying question
and marks opposing pairs. It does NOT assign the verdict: consensus, contested
and outlier are decided in code from cluster membership, so the labels are
deterministic and testable.

```
You are grouping claims drawn from several sources about a topic.

Topic: {topic}

Each claim is labelled with only its source id, so group by CONTENT, not by which
source is more authoritative. Group claims that address the SAME underlying
question, even when worded differently. For every group give a short topic_label
phrased as a question.

Then actively look for disagreement WITHIN each group. Two claims oppose when they
answer the same question differently in direction (creates vs destroys jobs),
magnitude (13% vs 40%), or timeframe (near-term vs long-run). Do not require the
wording to be mirror opposites; a claim that automation raises employment opposes
one that it lowers employment. List every such pair in opposing_pairs, and in
rationale name what the disagreement is about (direction, magnitude or timeframe).
Only leave opposing_pairs empty when the sources genuinely align.

Return JSON of the form:
{"clusters": [{"topic_label": "...?", "claim_ids": ["S1-C01", "S3-C01"],
  "opposing_pairs": [["S1-C01", "S3-C01"]], "rationale": "what the disagreement is about"}]}

Put every claim in exactly one group. Do not invent claim ids.

CLAIMS:
{claims}
```

Verdict rule applied in code after the call:

```
if any opposing pair in the cluster is from two different sources -> CONTESTED
elif the cluster spans two or more distinct sources                -> CONSENSUS
else                                                               -> OUTLIER
```

