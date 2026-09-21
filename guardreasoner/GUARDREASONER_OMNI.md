# GuardReasoner-Omni Eval — What Was Done

Getting [GuardReasoner-Omni](https://github.com/zzh-thu-22/GuardReasoner-Omni) (`zhu-thu-22/GuardReasoner-Omni-3B`, Qwen2.5-Omni based) running through the `cope-evaluation` harness (built for `zentropi-ai/cope-b-a4b` and `gpt-oss-safeguard-20b`), deployed on Modal, and evaluated end to end.

## 1. Audit: was the existing setup complete?

Before this work, `guardreasoner_adapter.py` and `serve_guardreasoner.py` already existed but had never been checked against the actual GuardReasoner-Omni source. Fetching the real `src/template.py`, `src/generate.py`, and the HF model card turned up three problems:

1. **Wrong prompt template.** The adapter used a 3-task template (`Request:` / `Completion: refusal-compliance` / `Response:`, no tags) — that's the *original* text-only GuardReasoner (`yueliu1999/GuardReasoner-3B`, Llama-3.2 based). The actual **Omni** line uses a 2-task template (`Request:` / `Response:` only) and requires the output to be wrapped in `<think>...</think>` then `<result>...</result>` — the model was fine-tuned against that exact instruction.
2. **Wrong model target.** `serve_guardreasoner.py` pointed at `yueliu1999/GuardReasoner-3B`, a different model family entirely from `zhu-thu-22/GuardReasoner-Omni-3B` (Qwen2.5-Omni based, multimodal).
3. **Not wired in.** `eval_cope.py` (the harness script) never imported or called the adapter at all — it was inert reference code. Nothing was actually runnable.

## 2. Fixes made

- **`guardreasoner_adapter.py`** — rewritten to match the verified Omni template verbatim: `OMNI_INSTRUCTION` (2-task, `<think>/<result>`), `INPUT_FORMAT`, `build_messages()` (system+user message split, `response="None"` convention for standalone-post moderation, matching the HF model card's own usage example), and `parse_verdict()` (regex matches `generate.py`'s own parsing logic, including its `none`/`harmless` synonyms for "unharmful").
- **`serve_guardreasoner.py`** — retargeted at `zhu-thu-22/GuardReasoner-Omni-3B`; added `--trust-remote-code` and `--limit-mm-per-prompt` (per the model card); confirmed via vLLM's model registry source that the checkpoint's `Qwen2_5OmniForConditionalGeneration` config auto-maps to vLLM's supported `Qwen2_5OmniThinkerForConditionalGeneration` implementation (no `--hf-overrides` needed); reused `serve_cope.py`'s already-proven Modal image (`nvidia/cuda:12.4.1-devel-ubuntu22.04`) instead of an untested one, so the model swap was the only new variable; renamed the Modal app/secret to `guardreasoner-omni` / `guardreasoner-omni-secrets` so it wouldn't collide with the existing `cope-b-a4b` deployment.
- **`cope-evaluation/eval/eval_guardreasoner.py`** (new) — closes the wiring gap: imports the fixed adapter, forces chat-mode requests with a CoT-sized `max_tokens` (2048 default), and writes the same `predictions_*.csv` / `summary_*.csv` shape as `eval_cope.py` so results are directly comparable.

All three were verified locally (message construction, parser output against synthetic `<think>/<result>` text, CLI/import/CSV-loading) before touching the live endpoint.

## 3. Deployment

- Modal CLI on `PATH` was outdated (0.65.66, rejected by the server); used the project venv's current client (1.5.3) instead.
- Created Modal secret `guardreasoner-omni-secrets` with a freshly generated `VLLM_API_KEY` (no `HF_TOKEN` needed — GuardReasoner-Omni is openly released).
- First deploy attempt failed: *"Please add a payment method to use L40S GPU functions"* — resolved once the user added one to the Modal account.
- Deployed successfully: `https://min-si--guardreasoner-omni-serve.modal.run`.

## 4. Smoke test, then full run

- **Smoke test** (2 rows, 1 policy, capped tokens): cold start ~318s, then real generation in ~2s/request. Output matched the verified template exactly — `<think>...reasoning...</think>` then `<result>Request:harmful Response:unharmful</result>` — and the parser extracted both verdicts with 0 errors. This confirmed the template rewrite was correct against live model output, not just against the repo source.
- **Full default sweep** (100 rows × 4 policies — `minimal`/`simple`/`medium`/`full`, self-harm test set), run while the container was still warm: 400/400 requests, **0 parse errors**, ~349s of generation.

### Results — GuardReasoner-Omni-3B, self-harm test set

| policy  | n   | precision | recall | f1    | accuracy |
|---------|-----|-----------|--------|-------|----------|
| minimal | 100 | 0.880     | 0.440  | 0.587 | 0.690    |
| simple  | 100 | 0.939     | 0.620  | 0.747 | 0.790    |
| medium  | 100 | 0.907     | 0.780  | 0.839 | 0.850    |
| full    | 100 | 0.929     | 0.780  | 0.848 | 0.860    |

Same pattern seen with cope-b and gpt-oss-safeguard in `cope-evaluation/RESULTS.md`: more policy detail sharply improves recall (0.44 → 0.78) while precision stays high throughout.

Output files:
- `cope-evaluation/eval/results/predictions_guardreasoner_selfharm_20260819_174946.csv` (per-row, with full `<think>` traces)
- `cope-evaluation/eval/results/summary_guardreasoner_selfharm_20260819_174946.csv`

## 5. Cost

User's ceiling: **$10 for the eval.**

- Smoke test: ~$0.17 metered
- Full 400-request sweep (container reused, no second cold start): brought cumulative metered cost to **$1.21**
- Fully offset by Modal's free-tier credits → **$0.00 billed**
- App explicitly stopped (`modal app stop guardreasoner-omni --yes`) immediately after, terminating all 7 running containers, confirmed via `modal app list` (state: `stopped`, 0 tasks) — no further idle billing.

## 6. Current state / how to continue

- Modal app `guardreasoner-omni` is **stopped**. Redeploying (`modal deploy serve_guardreasoner.py`) will trigger a fresh image build (cached, fast) and a new cold start (~5 min) on first request.
- The `guardreasoner-omni-secrets` Modal secret holds a `VLLM_API_KEY` — the value isn't retrievable from Modal after creation; if the local copy is lost, regenerate and overwrite with `modal secret create guardreasoner-omni-secrets VLLM_API_KEY=<new-key> --force` (this doesn't require redeploying, since no container was running).
- To run the **sexual-content domain** next:
  ```bash
  python eval_guardreasoner.py \
    --endpoint https://min-si--guardreasoner-omni-serve.modal.run/v1/chat/completions \
    --test-set sex_eval/test_set.csv \
    --label guardreasoner_sex \
    --policies sexual_content_minimal sexual_content_simple sexual_content_medium sexual_content_zentropi_long sexual_content_oai
  ```
  (redeploy first — the app is currently stopped). Expected cost: another cold start (~$0.17) plus generation, trivial against the $10 budget.
