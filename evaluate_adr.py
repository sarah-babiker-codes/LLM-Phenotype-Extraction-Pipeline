"""
Evaluate the ADR reasoning-LLM prompts (adr1, adr2) against the derived
ADR gold standard.

Kept fully separate from evaluate_llm.py (the clozapine pipeline): this
script has its own extract_answer, extract, and merge functions rather
than importing them, so that changes here never affect the already
validated clozapine1/clozapine2 results.

adr_present is a derived variable (see code_gold_standard_adr.py), not
a raw CardiffCOGS column, so the gold standard is read from
gold_standard_adr.csv rather than the .sav file.

Usage:
    python evaluate_adr.py --prompt adr1 adr2
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
)

PROJECT_DIR = Path("/scratch/SCWF00151/c.c25097054")
DATA_DIR = PROJECT_DIR / "treatment-response"
RESULTS_DIR = PROJECT_DIR / "results"
GOLD_STANDARD_CSV = RESULTS_DIR / "gold_standard_adr.csv"

ID_COL = "Study_ID"
RAW_COL = "Generated_text"
PRED_COL = "Extract"
RAW_FILE = "cogs_predict_tr.csv"
THINK_TAG = "</think>"


def extract_answer(text):
    """Extract the model's final binary (0/1) answer from a reasoning trace.

    Text after the </think> tag is searched, in order of priority, for:
      1. an explicit "Answer:" or "Response:" label
      2. a digit alone on its own line
      3. a digit at the start of a line followed by punctuation
         (e.g. "0. explanation...", "1 - explanation...")
      4. a digit alone in parentheses (e.g. "...ADRs (0).")
      5. a digit at the start of a line followed by a word, with no
         punctuation (e.g. "0 The patient has no documented...")

    Each priority is searched with re.findall, and the LAST match is
    taken when more than one is found, on the basis that a model's
    final stated answer should take precedence over any earlier,
    provisional statement in its reasoning trace. This is applied
    consistently across all five priorities, since these prompts
    frequently produce a "SUMMARY:" section with the answer embedded
    in a sentence rather than isolated on its own line.

    Returns NA if no answer can be identified; a small residual of
    non-standard formatting is expected and left unresolved.
    """
    if not isinstance(text, str):
        return pd.NA
    tail = text.split(THINK_TAG)[-1] if THINK_TAG in text else text

    patterns = [
        r"(?:answer|response):?\s*\**\s*([01])\b",
        r"^\s*\**([01])\**\s*$",
        r"^\s*\**([01])\**\s*[\.\-:|;,/]",
        r"\(([01])\)",
        r"^\s*\**([01])\**\s+[A-Za-z]",
    ]
    for pattern in patterns:
        flags = re.IGNORECASE | re.MULTILINE
        matches = re.findall(pattern, tail, flags)
        if matches:
            return int(matches[-1])

    return pd.NA


def extract(prompt):
    src = DATA_DIR / f"dsq_{prompt}" / RAW_FILE
    df = pd.read_csv(src)
    df[PRED_COL] = df[RAW_COL].apply(extract_answer)
    out = df[[ID_COL, PRED_COL]]

    RESULTS_DIR.mkdir(exist_ok=True)
    dest = RESULTS_DIR / f"{prompt}_extracted.csv"
    out.to_csv(dest, index=False)

    print(f"[{prompt}] extracted {len(out)} rows | "
          f"0: {(out[PRED_COL] == 0).sum()} | "
          f"1: {(out[PRED_COL] == 1).sum()} | "
          f"no answer: {out[PRED_COL].isna().sum()}")
    return out


def merge(prompt):
    gold = pd.read_csv(GOLD_STANDARD_CSV)[[ID_COL, "adr_present"]].copy()

    llm = pd.read_csv(RESULTS_DIR / f"{prompt}_extracted.csv")
    merged = llm.merge(gold, on=ID_COL, how="left")

    dest = RESULTS_DIR / f"{prompt}_merged.csv"
    merged.to_csv(dest, index=False)

    print(f"[{prompt}] merged on {ID_COL} against adr_present | "
          f"{len(merged)} rows | "
          f"matched gold: {merged['adr_present'].notna().sum()} | "
          f"no gold: {merged['adr_present'].isna().sum()}")
    return merged


def evaluate(prompt):
    m = pd.read_csv(RESULTS_DIR / f"{prompt}_merged.csv")
    n_total = len(m)
    m = m.dropna(subset=[PRED_COL, "adr_present"])

    y_true = m["adr_present"].astype(int)
    y_pred = m[PRED_COL].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "prompt": prompt,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "n_scorable": len(m),
        "n_total": n_total,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def run(prompts):
    results = []
    for prompt in prompts:
        extract(prompt)
        merge(prompt)
        results.append(evaluate(prompt))

    df = pd.DataFrame(results)
    cols = ["prompt", "macro_f1", "accuracy", "balanced_accuracy",
            "specificity", "sensitivity", "n_scorable"]
    print("\n" + "=" * 70)
    print(df[cols].to_string(index=False, float_format="%.3f"))

    print("\nconfusion matrices (gold = rows, LLM = cols)")
    for r in results:
        print(f"\n[{r['prompt']}]  scorable {r['n_scorable']} of {r['n_total']}")
        print(f"            LLM=0   LLM=1")
        print(f"  gold=0   {r['tn']:6d}  {r['fp']:6d}")
        print(f"  gold=1   {r['fn']:6d}  {r['tp']:6d}")

    dest = RESULTS_DIR / "evaluation_metrics_adr.csv"
    df.to_csv(dest, index=False)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, nargs="+",
                   help="one or more of: adr1 adr2")
    a = p.parse_args()
    run(a.prompt)
