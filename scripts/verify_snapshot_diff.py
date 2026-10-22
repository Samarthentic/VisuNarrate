#!/usr/bin/env python3
"""Verify exact text invariance between regression snapshots."""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEFORE_PATH = os.path.join(BASE_DIR, "data", "evaluation_results", "snapshot_before_final.json")
AFTER_PATH = os.path.join(BASE_DIR, "data", "evaluation_results", "snapshot_after_final.json")

def main():
    if not os.path.exists(BEFORE_PATH):
        print(f"Error: {BEFORE_PATH} not found.")
        sys.exit(1)
    if not os.path.exists(AFTER_PATH):
        print(f"Error: {AFTER_PATH} not found.")
        sys.exit(1)

    with open(BEFORE_PATH, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(AFTER_PATH, "r", encoding="utf-8") as f:
        after = json.load(f)

    if len(before) != len(after):
        print(f"Length mismatch: {len(before)} vs {len(after)}")
        sys.exit(1)

    mismatches = []
    for i, (b, a) in enumerate(zip(before, after)):
        fn = b.get("filename", f"Item #{i+1}")
        b_narr = b.get("narration")
        a_narr = a.get("narration")
        b_pred = b.get("prediction")
        a_pred = a.get("prediction")
        b_hyps = [h.get("text") for h in b.get("hypotheses", [])]
        a_hyps = [h.get("text") for h in a.get("hypotheses", [])]

        diffs = []
        if b_narr != a_narr:
            diffs.append(f"Narration: '{b_narr}' != '{a_narr}'")
        if b_pred != a_pred:
            diffs.append(f"Prediction: '{b_pred}' != '{a_pred}'")
        if b_hyps != a_hyps:
            diffs.append(f"Hypotheses: {b_hyps} != {a_hyps}")

        if diffs:
            mismatches.append((fn, diffs))

    print(f"==================================================")
    print(f"REGRESSION INVARIANCE VERIFICATION (17 Benchmark Images)")
    print(f"==================================================")
    if not mismatches:
        print("✅ 100% INVARIANCE MATCH: All narration and prediction texts are strictly identical!")
        print(f"Verified {len(before)} benchmark images.")
        print(f"==================================================")
        sys.exit(0)
    else:
        print(f"❌ {len(mismatches)} MISMATCHES DETECTED:")
        for fn, diffs in mismatches:
            print(f"  - {fn}:")
            for d in diffs:
                print(f"      {d}")
        print(f"==================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
