#!/usr/bin/env python3
"""
Build the privacy test set for the CoPE evaluation harness.

This is the Part C step (merge) from the sex_eval pattern. There is no Part A
here — see README.md — because the source corpus (PGB v1) is fully synthetic by
design: every identifier is a canary in a reserved namespace, so the test set
contains no real personal data.

Input:  PGB v1 corpus JSONL (labelled, 1,200 records)
Output: test_set.csv           id, content, ground_truth   (gold_test, 300 rows)
        test_set_full.csv      id, content, ground_truth   (all 1,200 rows)
        test_set_metadata.csv  id + every slicing field, for error analysis

`ground_truth` is 1 when the AI assistant's response in `content` discloses
protected information, 0 when it does not. It is a judgement about the
**response**, not the request — see README.md, "What the label means".

Usage:
    python build_test_set.py --corpus ../../../pgb_v1/pgb_v1_corpus.jsonl
    python build_test_set.py --corpus <path> --split all --out test_set_full.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# The exact block a response-level guard expects. Keeping the request in the
# content is deliberate: a privacy disclosure can only be judged against what
# was asked.
CONTENT = "Human user:\n{prompt}\n\nAI assistant:\n{response}\n\n"

# Tier mapping, mirroring the strong-signal / borderline / no-signal split the
# sex_eval sampler produces. PGB's response arms give this for free.
TIER = {
    "comply_leak":      "strong_signal",   # blatant disclosure — drives recall
    "comply_overshare": "strong_signal",   # volunteered disclosure on a benign ask
    "refuse_leak":      "borderline",      # refuses, then discloses anyway
    "comply_safe":      "borderline",      # complies with a harmful ask, discloses nothing
    "refuse_clean":     "no_signal",       # correct refusal
    "refuse_over":      "no_signal",       # over-refusal — drives precision
    "comply_helpful":   "no_signal",       # ordinary helpful answer
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="PGB v1 corpus JSONL")
    ap.add_argument("--split", default="gold_test",
                    choices=["gold_test", "train", "val", "all"])
    ap.add_argument("--out", default="test_set.csv")
    ap.add_argument("--metadata-out", default="test_set_metadata.csv")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.corpus, encoding="utf-8")]
    if args.split != "all":
        recs = [r for r in recs if r["metadata"].get("split") == args.split]
    if not recs:
        raise SystemExit(f"no records for split={args.split}")

    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "content", "ground_truth"])
        for r in recs:
            w.writerow([
                r["record_id"],
                CONTENT.format(prompt=r["user_input"], response=r["agent_response"]),
                r["metadata"]["bin_response_harmful"],
            ])

    meta = Path(args.metadata_out)
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tier", "response_arm", "attack_category", "attack_vector",
                    "task_family", "severity", "adversary_tier", "domain",
                    "request_is_harmful", "response_is_refusal", "over_refusal",
                    "ground_truth_notes"])
        for r in recs:
            m = r["metadata"]
            w.writerow([
                r["record_id"], TIER[m["response_arm"]], m["response_arm"],
                r["attack_category"], r["attack_vector"], m["pgb_task_family"],
                r["severity"], m["adversary_tier"], r["domain"],
                int(m["gt_prompt_harmfulness"] == "harmful"),
                int(m["gt_refusal"] == "refusal"),
                int(m["over_refusal"]),
                r["ground_truth_notes"],
            ])

    gt = Counter(r["metadata"]["bin_response_harmful"] for r in recs)
    tiers = Counter(TIER[r["metadata"]["response_arm"]] for r in recs)
    print(f"wrote {out} — {len(recs)} rows")
    print(f"  ground_truth: 1={gt[1]}  0={gt[0]}")
    print(f"  tiers: {dict(tiers)}")
    print(f"wrote {meta} — join on `id` for per-slice error analysis")


if __name__ == "__main__":
    main()
