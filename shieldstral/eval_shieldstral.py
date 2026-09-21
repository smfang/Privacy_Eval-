"""
Evaluate mistralai/Shieldstral-1.0-3B across multiple policy prompts of varying
detail, reusing this repo's harness shape (test set, --policies axis, results CSVs).

Shieldstral's interface (see shieldstral_adapter.py) is a single yes/no question
per call: the policy goes in <Instruct>, a FIXED question goes in <Query>, and
the content goes in <Document> completely verbatim — no request/response
splitting is needed, even for pre-formatted pair content like the privacy set,
because you choose what the question is about by writing the query itself.

Usage:
    export VLLM_API_KEY=sk-...                          # the key from `modal secret create shieldstral-secrets`

    # self-harm (default test set + policies, standalone posts)
    python eval_shieldstral.py --endpoint https://<your-modal-url>/v1/chat/completions

    # smoke test
    python eval_shieldstral.py --endpoint ... --limit 5 --policies minimal

    # privacy set (request/response pairs; --which response asks about the
    # response specifically, no splitting needed)
    python eval_shieldstral.py --endpoint ... \\
        --test-set ../guardreasoner/privacy_eval/test_set.csv \\
        --policy-dir ../guardreasoner/policies \\
        --label shieldstral_privacy \\
        --policies privacy_minimal privacy_simple privacy_medium privacy_full \\
        --which response

Outputs (written to cope-evaluation/eval/results/, alongside every other model's runs):
    predictions_<label>_<timestamp>.csv
    summary_<label>_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

HERE = Path(__file__).parent
COPE_EVAL_DIR = HERE.parent / "cope-evaluation" / "eval"
POLICIES_DIR = COPE_EVAL_DIR / "policies"
RESULTS_DIR = COPE_EVAL_DIR / "results"
DEFAULT_TEST_SET = COPE_EVAL_DIR / "test_set.csv"

from shieldstral_adapter import build_messages, parse_verdict, STANDALONE_QUERY, RESPONSE_QUERY  # noqa: E402

DEFAULT_MODEL = "mistralai/Shieldstral-1.0-3B"


def load_policies(names: list[str], policy_dir: Path) -> dict[str, str]:
    out = {}
    for name in names:
        path = policy_dir / f"{name}.md"
        if not path.exists():
            sys.exit(f"missing policy file: {path}")
        out[name] = path.read_text()
    return out


def load_test_set(path: Path, limit: int | None) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("ground_truth", "").strip() in ("0", "1")]
    return rows[:limit] if limit else rows


def call_shieldstral(policy: str, content: str, query: str, api_key: str, endpoint: str, model: str,
                      max_tokens: int, timeout: int = 120) -> tuple[str, str]:
    messages = build_messages(policy, content, query)
    body = {"model": model, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}
    resp = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"] or ""
    pred_int, ok = parse_verdict(raw)
    pred = str(pred_int) if ok else ""
    return pred, raw


def warmup(api_key: str, endpoint: str, model: str, query: str, max_tokens: int) -> None:
    print(f"warming up {endpoint}...", flush=True)
    t0 = time.time()
    pred, raw = call_shieldstral("Flag harmful content.", "hello", query, api_key, endpoint, model,
                                  max_tokens=max_tokens, timeout=300)
    print(f"  warm in {time.time()-t0:.1f}s, pred={pred!r}, raw={raw[:200]!r}", flush=True)


def run_one(args):
    row_id, content, policy_name, policy_text, query, api_key, endpoint, model, max_tokens = args
    try:
        pred, raw = call_shieldstral(policy_text, content, query, api_key, endpoint, model,
                                      max_tokens=max_tokens)
        return row_id, policy_name, pred, raw, None
    except Exception as e:
        return row_id, policy_name, "", "", str(e)


def metrics(preds: list[str], truth: list[str]) -> dict:
    tp = sum(1 for p, t in zip(preds, truth) if p == "1" and t == "1")
    fp = sum(1 for p, t in zip(preds, truth) if p == "1" and t == "0")
    fn = sum(1 for p, t in zip(preds, truth) if p == "0" and t == "1")
    tn = sum(1 for p, t in zip(preds, truth) if p == "0" and t == "0")
    errors = sum(1 for p in preds if p == "")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(preds) - errors, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, errors=errors,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET,
                    help="path to CSV with columns id, content, ground_truth")
    ap.add_argument("--label", default="shieldstral",
                    help="label prefix for output files (e.g., 'shieldstral_privacy')")
    ap.add_argument("--limit", type=int, default=None, help="limit number of rows")
    ap.add_argument("--policies", nargs="+", default=["minimal", "simple", "medium", "full"])
    ap.add_argument("--policy-dir", type=Path, default=None,
                    help=f"override where policy .md files are read from (default: {POLICIES_DIR})")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Shieldstral is single-token, non-reasoning — can run higher concurrency than the Omni harness")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--endpoint", required=True,
                    help="vLLM /v1/chat/completions URL for your deployed Shieldstral endpoint")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="Shieldstral emits exactly one yes/no token — no CoT budget needed")
    ap.add_argument("--which", choices=["request", "response"], default="request",
                    help="request: ask a standalone-post question (STANDALONE_QUERY). "
                         "response: ask about the AI's response in a pair (RESPONSE_QUERY). "
                         "Overridden by --query if given.")
    ap.add_argument("--query", default=None,
                    help="override the fixed yes/no question sent as <Query> (advanced — see "
                         "shieldstral_adapter.py's docstring on why this should stay constant "
                         "across policy variants within one run)")
    args = ap.parse_args()

    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        sys.exit("set VLLM_API_KEY env var (the key you used in `modal secret create shieldstral-secrets`)")

    query = args.query or (RESPONSE_QUERY if args.which == "response" else STANDALONE_QUERY)

    policy_dir = args.policy_dir or POLICIES_DIR
    policies = load_policies(args.policies, policy_dir)
    rows = load_test_set(args.test_set, args.limit)
    print(f"loaded {len(rows)} test rows from {args.test_set}, {len(policies)} policies from {policy_dir}: {list(policies)}")
    print(f"query (<Query>, fixed across all policies this run): {query!r}")

    if not args.skip_warmup:
        warmup(api_key, args.endpoint, args.model, query, args.max_tokens)

    jobs = []
    for row in rows:
        for pname, ptext in policies.items():
            jobs.append((row["id"], row["content"], pname, ptext, query, api_key, args.endpoint, args.model,
                         args.max_tokens))

    print(f"firing {len(jobs)} requests with concurrency={args.concurrency}...", flush=True)
    results: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            row_id, pname, pred, raw, err = fut.result()
            results[(row_id, pname)] = (pred, raw, err)
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"  {completed}/{len(jobs)} ({rate:.2f}/s, {elapsed:.0f}s elapsed)", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.label}_{ts}"
    pred_path = RESULTS_DIR / f"predictions_{suffix}.csv"

    policy_names = list(policies)
    header = ["id", "content", "ground_truth"]
    for p in policy_names:
        header += [f"{p}_pred", f"{p}_raw"]
    header += [f"{p}_correct" for p in policy_names]

    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            line = [row["id"], row["content"], row["ground_truth"]]
            for p in policy_names:
                pred, raw, err = results[(row["id"], p)]
                line += [pred, raw if not err else f"ERROR: {err}"]
            for p in policy_names:
                pred, _, _ = results[(row["id"], p)]
                line.append("1" if pred == row["ground_truth"] else "0")
            w.writerow(line)
    print(f"\nwrote predictions: {pred_path}")

    summary_path = RESULTS_DIR / f"summary_{suffix}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n", "tp", "fp", "fn", "tn", "errors",
                    "precision", "recall", "f1", "accuracy"])
        print(f"\n{'policy':16s} {'n':>4s} {'tp':>4s} {'fp':>4s} {'fn':>4s} {'tn':>4s} "
              f"{'err':>4s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'acc':>6s}")
        print("-" * 76)
        for p in policy_names:
            preds = [results[(row["id"], p)][0] for row in rows]
            truth = [row["ground_truth"] for row in rows]
            m = metrics(preds, truth)
            w.writerow([p, len(rows), m["tp"], m["fp"], m["fn"], m["tn"], m["errors"],
                        f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                        f"{m['f1']:.3f}", f"{m['accuracy']:.3f}"])
            print(f"{p:16s} {len(rows):>4d} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d} {m['tn']:>4d} "
                  f"{m['errors']:>4d} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['accuracy']:>6.3f}")
    print(f"\nwrote summary: {summary_path}")


if __name__ == "__main__":
    main()
