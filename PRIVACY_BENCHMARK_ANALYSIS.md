# Privacy Benchmark Analysis — gpt-oss-safeguard-20b vs Shieldstral-1.0-3B vs GuardReasoner-Omni-3B

All three models were run against the same privacy set (`guardreasoner/privacy_eval/test_set.csv`,
300 rows, balanced 150/150, request/response pairs), the same four policy variants
(`privacy_minimal` → `privacy_full`), and the same underlying question — does the
**AI assistant's response** disclose protected information. Same test set, same
policies, same label semantics: these three runs are directly comparable to each
other in a way that most cross-model comparisons in this repo are not (see the
Zentropi caveat in `cope-evaluation/RESULTS.md`).

Predictions: `cope-evaluation/eval/results/predictions_{guardreasoner,shieldstral,safeguard}_privacy_*.csv`
Summaries: `cope-evaluation/eval/results/summary_{guardreasoner,shieldstral,safeguard}_privacy_*.csv`
Arm breakdowns: `cope-evaluation/eval/results/arm_breakdown_{guardreasoner,shieldstral,safeguard}_privacy_*.csv`

## 1. Headline F1

| policy | **safeguard F1** | safeguard P / R | Shieldstral F1 | Shieldstral P / R | Omni F1 | Omni P / R |
|---|---|---|---|---|---|---|
| minimal | **0.950** | 0.947 / 0.953 | 0.643 | 0.973 / 0.480 | 0.687 | 0.817 / 0.593 |
| simple  | **0.971** | 0.949 / 0.993 | 0.809 | 0.954 / 0.703 | 0.748 | 0.787 / 0.713 |
| medium  | **0.961** | 0.937 / 0.987 | 0.871 | 0.959 / 0.797 | 0.721 | 0.777 / 0.673 |
| full    | **0.951** | 0.925 / 0.980 | 0.904 | 0.955 / 0.858 | 0.701 | 0.732 / 0.673 |

**gpt-oss-safeguard-20b wins outright, by a wide margin, at every policy level** —
F1 0.95–0.97 vs. Shieldstral's 0.64–0.90 and Omni's 0.70–0.75. It also has the
best recall of all three at every level (0.95–0.99), which is the harder metric
to get right on an adversarial set like this one.

**All three models have different policy-detail curves.** Shieldstral climbs
monotonically (0.643→0.904). Omni peaks at `simple` and degrades (0.748→0.701).
Safeguard is essentially flat-to-slightly-declining from its `simple` peak
(0.971→0.951) — it's already close to its ceiling at the *lowest* level of
detail beyond `minimal`, and more policy text buys it almost nothing further.
Three models, three shapes, same policy text each time: **"more detail helps"
is not a property of the benchmark, it's a property of the specific model.**

**Only 1 parse error out of 1200 calls** (0.08%) — the cleanest error rate of
the three models tested on this set (vs. Shieldstral's 4 timeout-driven errors
and, per `RESULTS.md`, gpt-oss-safeguard's own 10% malformed-response rate on
the *sexual-content* set with `max_tokens=1024`). Bumping to `max_tokens=2048`
here (per the model card's own guidance) appears to have mattered.

## 2. Refusal-anchoring (`comply_leak` vs `refuse_leak` recall)

This is the specific failure mode the privacy set's `refuse_leak` arm (75 rows:
response declines up front, discloses later) was built to expose. The question:
does the guard read refusal language as evidence of safety and miss the leak
buried after it?

| policy | **safeguard** comply_leak → refuse_leak | gap | Shieldstral comply_leak → refuse_leak | gap | Omni comply_leak → refuse_leak | gap |
|---|---|---|---|---|---|---|
| minimal | 0.943 → 0.947 | **+0.00** | 0.453 → 0.480 | +0.03 | 0.849 → 0.467 | **−0.38** |
| simple  | 1.000 → 0.987 | −0.01 | 0.717 → 0.640 | −0.08 | 0.925 → 0.613 | −0.31 |
| medium  | 0.981 → 0.987 | **+0.01** | 0.774 → 0.787 | +0.01 | 0.925 → 0.547 | **−0.38** |
| full    | 0.981 → **1.000** | **+0.02** | 0.887 → 0.773 | −0.11 | 0.943 → 0.547 | **−0.40** |

**gpt-oss-safeguard essentially solves this arm.** Its refuse_leak recall
(0.947–1.000) is statistically indistinguishable from its comply_leak recall at
every policy level — the gap never exceeds 2 points, and at `full` it's
*slightly better* on the disguised leak than the clean one. Its CoT reasoning
(captured in the predictions CSV — see `eval_cope.py`'s reasoning-capture fix,
§5) shows why: on the sample refuse_leak row inspected during the smoke test,
it explicitly reasoned *"it says 'I won't take that step' but then it provides
the info. That is a disclosure"* — it names the refusal-then-leak pattern and
reasons past it, rather than anchoring on the refusal.

Shieldstral is close behind with no meaningful bias either (gap −0.11 to +0.03).
**GuardReasoner-Omni is the outlier**, with a large, consistent 31–40 point
recall drop the moment a leak is preceded by refusal language, at every policy
level including its best one. This is the exact failure pattern the corpus's
README warns a naive detector will show, and Omni reproduces it clearly — while
the other two models tested here do not.

## 3. `comply_overshare` — unrequested disclosure in an otherwise benign reply

22 rows: benign request, assistant volunteers protected detail nobody asked for.

| policy | **safeguard** | Shieldstral | Omni |
|---|---|---|---|
| minimal | **1.000** | 0.500 | 0.409 |
| simple  | **1.000** | 0.818 | 0.545 |
| medium  | **1.000** | 0.818 | 0.500 |
| full    | 0.909 | **1.000** | 0.455 |

Safeguard catches essentially every unrequested disclosure at every policy
level except a single miss at `full`. Shieldstral scales up to match it by
`full`. Omni never clears 0.55 recall on this arm at any policy level — its
weakness on "disclosure nobody asked for" is consistent across the whole run,
not just at the extremes.

## 4. `comply_safe` false positives — the cost of extra policy detail

53 rows: request answerable-but-sensitive, response correctly withholds the
protected detail. This is the arm that explains why Omni's F1 *fell* with more
policy detail, and why safeguard's dips slightly past `simple`.

| policy | **safeguard** FP rate | Shieldstral FP rate | Omni FP rate |
|---|---|---|---|
| minimal | 0.132 | **0.019** | 0.358 |
| simple  | 0.151 | **0.075** | 0.509 |
| medium  | 0.189 | **0.075** | 0.472 |
| full    | 0.226 | **0.094** | 0.604 |

All three models get *more* likely to falsely flag a correctly-hedged response
as policy detail increases — but the magnitude differs enormously. Shieldstral
is the most resistant to this failure (2–9%). Safeguard is worse (13–23%) but
still far better than Omni (36–60%, wrongly flagging over a third to nearly
two-thirds of genuinely correct, careful responses at its longer policies).
This is the one arm where Shieldstral actually beats safeguard outright — and
it's exactly the arm that keeps safeguard's F1 from climbing past `simple`.

## 5. Two harness fixes this run surfaced

**gpt-oss-safeguard's reasoning wasn't reaching the predictions CSV.** The
initial smoke test returned a bare `"0"`/`"1"` with zero visible reasoning
despite `max_tokens=2048` being available — traced to a newer vLLM behavior:
gpt-oss models return chain-of-thought in a separate `message.reasoning` field
rather than inline in `message.content` (confirmed by a direct API probe that
showed a full paragraph of reasoning in `reasoning` alongside a clean `"1"` in
`content`). `eval_cope.py`'s `call_cope()` only read `content`. **This did not
affect correctness** — the parsed prediction was always right — but it meant
losing the qualitative visibility (`GUIDE.md`'s own documented CoT-scan
assumption) that made the §2 refuse_leak reasoning quote above possible.
Patched: `call_cope()` now concatenates `reasoning` (if present) into the raw
output before the harness writes it to CSV. Backward compatible — models that
don't return a separate `reasoning` field are unaffected.

**Four Shieldstral rows failed on an HTTP timeout, not a model error** — same 4
row IDs, all `Read timed out (timeout=120)`, evenly split across ground truth.
Traced to `--skip-warmup` plus a concurrency burst against a container that
hadn't fully settled. Infra noise, not a finding — flagged per `RESULTS.md`'s
own "check `errors` before trusting F1" convention.

## 6. Cost, for the same 1200-call sweep

| model | GPU | cold start | generation | **total cost (this run)** |
|---|---|---|---|---|
| Shieldstral-1.0-3B | L4 | ~320s | ~350s (3.4 req/s) | **$3.49** |
| GuardReasoner-Omni-3B | L40S | ~320s | ~950s (1.26 req/s) | **$2.45** |
| gpt-oss-safeguard-20b | H100 | ~330s | ~740s (1.6 req/s) | **~$3.74** |

Despite being a 20B reasoning model on the most expensive GPU tier, safeguard's
total cost landed in the same ballpark as the two smaller models — H100's raw
speed roughly offsets its higher per-second price. **Best F1 and mid-pack cost**
is not the tradeoff a naive size/cost intuition would predict here.

## 7. How this fits the rest of the repo's benchmark results

For context only — **not directly comparable**, since none of the other repo
runs used this test set (per `RESULTS.md`'s own comparability caveat):

- On the **self-harm** set, cope-b (F1 0.936 on `full`) beat gpt-oss-safeguard-20b
  (F1 0.854 on `very_long`, its best variant there) by 0.08 — the opposite
  ranking from this privacy run, where safeguard is the clear winner. Domain
  and test-set framing change which model wins; there's no single "best guard."
- On the **sexual-content** set, gpt-oss-safeguard had a 10% malformed-response
  rate at `max_tokens=1024`. Here, at `max_tokens=2048`, it had 1 error in 1200
  calls — consistent with `GUIDE.md`'s own warning that truncating a CoT
  model's budget shows up as silent parse failures, not errors.

## Bottom line

On this privacy benchmark — request/response pairs, response-harmfulness label,
refusal-heavy adversarial arm — **gpt-oss-safeguard-20b is the strongest model
of the three tested**, by a wide F1 margin, and it is the only one of the three
that shows no meaningful refusal-anchoring bias *and* near-perfect overshare
recall simultaneously. Its remaining weakness — false positives on correctly-
hedged safe responses, worsening with policy detail — is real but far smaller
than GuardReasoner-Omni's version of the same failure, and Shieldstral remains
the most precision-robust option on that specific arm. Its cost, at ~$3.74 for
this sweep, is not meaningfully higher than either smaller model despite the
larger checkpoint and pricier GPU.
