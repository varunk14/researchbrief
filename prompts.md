# Prompts

The two model-facing stages and the prompts they use. Both stages run at
temperature 0 with JSON output and one retry that feeds the parse error back.

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

## Stage 4 - Cross-source synthesis

Documented in this file when the synthesis stage is built.
