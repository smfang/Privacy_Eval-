# GuardReasoner-Omni-3B vs Shieldstral-1.0-3B — Research & Results Summary

## What these models are

**GuardReasoner-Omni-3B** (`zhu-thu-22/GuardReasoner-Omni-3B`, Qwen2.5-Omni
based) is a reasoning-based multimodal guardrail. It's fine-tuned to answer a
fixed 2-task instruction — is the human's *request* harmful, is the AI's
*response* harmful — via explicit chain-of-thought wrapped in `<think>` tags,
followed by a `<result>` block giving both verdicts. It has no notion of a
custom policy built in; a policy can only be injected as extra context
prepended to its fixed instruction, which steers it partially.

**Shieldstral-1.0-3B** (`mistralai/Shieldstral-1.0-3B`, Ministral-3-3B based)
is a policy-*adaptive* classifier with the opposite design: it has no built-in
notion of "harmful" at all. Every judgment is a single yes/no question,
answered against a document, both supplied at inference time — three fields:
`<Instruct>` (context/policy), `<Query>` (one fixed yes/no question),
`<Document>` (the content). Output is a single yes/no token, no reasoning
chain. Same parameter count as Omni, same open license (Apache 2.0), much
smaller GPU footprint (fits 16GB bf16 vs. Omni's larger multimodal-tower
requirement).

## What was tested

Both models were run against the same privacy benchmark (300 rows, balanced
150/150, `guardreasoner/privacy_eval/test_set.csv`) — synthetic request/AI-response
pairs where the label is whether the **response** discloses protected
information about an identifiable person. Same four policy variants
(`privacy_minimal` → `privacy_full`, increasing detail), same underlying
question, deployed via Modal + vLLM.

Mapping the benchmark onto each model required different adapter work:
- **Omni** needed a request/response *splitter* — its template re-wraps
  whatever content you pass it, so the pre-formatted pair had to be split
  apart first, or it would end up double-wrapped with a contradictory second
  "AI assistant: None" appended.
- **Shieldstral** needed no such split — `<Document>` is used completely
  verbatim, and you choose what the yes/no question is about by writing the
  query itself. Its native design fit this benchmark's pair format more
  directly.

## Headline results

| policy | Shieldstral F1 | Shieldstral P / R | Omni F1 | Omni P / R |
|---|---|---|---|---|
| minimal | 0.643 | 0.973 / 0.480 | 0.687 | 0.817 / 0.593 |
| simple  | 0.809 | 0.954 / 0.703 | 0.748 | 0.787 / 0.713 |
| medium  | 0.871 | 0.959 / 0.797 | 0.721 | 0.777 / 0.673 |
| full    | **0.904** | 0.955 / 0.858 | 0.701 | 0.732 / 0.673 |

**Opposite policy-detail curves.** Shieldstral climbs monotonically with more
policy detail (0.643 → 0.904). Omni peaks early, at `simple`, then *degrades*
(0.748 → 0.701) as policies get longer. Same policy text, same test set —
"more detail helps" turned out to be a property of the model, not the
benchmark.

**Shieldstral wins from `medium` onward**, despite being the architecturally
simpler model — single forward pass, one token, vs. Omni's full reasoning
trace. Reasoning capacity didn't translate into an advantage here.

## The key behavioral finding: refusal-anchoring

The test set includes a specific adversarial arm (`refuse_leak`, 75 rows):
the AI response declines up front, then discloses the protected detail later
in the same reply. This targets a known failure mode — a guard that reads
refusal language as evidence of safety will miss the leak buried after it.

| policy | Shieldstral: clean-leak → disguised-leak recall | gap | Omni: clean-leak → disguised-leak recall | gap |
|---|---|---|---|---|
| minimal | 0.453 → 0.480 | +0.03 | 0.849 → 0.467 | **−0.38** |
| full    | 0.887 → 0.773 | −0.11 | 0.943 → 0.547 | **−0.40** |

**GuardReasoner-Omni shows a large, consistent refusal-anchoring bias** —
recall drops 31–40 points at every policy level the instant the same leak is
preceded by refusal framing. **Shieldstral shows no comparable bias** (gap
stays within ±0.11, sign-flipping). Whatever Shieldstral keys on to judge
disclosure, it is not primarily "did the response start by refusing."

A second, related pattern: Omni's false-positive rate on *correctly-hedged
safe* responses climbs sharply with policy detail (36% → 60% wrongly flagged
at `minimal` → `full`), while Shieldstral's stays low throughout (2% → 9%).
This is what explains Omni's inverted F1 curve above — its longer policies
make it pattern-match on language the policy itself uses as an example of
*safe* handling, and flag it anyway.

## Bottom line

On this benchmark, **Shieldstral-1.0-3B is the stronger model**: higher F1
from `simple` onward, consistently high precision, and no measurable
refusal-anchoring bias — at a fraction of the inference cost (single token vs.
full chain-of-thought) and a smaller GPU footprint. GuardReasoner-Omni's
weakness is specific and reproducible, not diffuse: it anchors on refusal
language, and giving it more policy detail makes that worse, not better.

*(A third model, gpt-oss-safeguard-20b, was also run on this same benchmark
and outperforms both — see `PRIVACY_BENCHMARK_ANALYSIS.md` for the full
three-way comparison, arm-by-arm breakdowns, and cost figures.)*
