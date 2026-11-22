"""
Evaluate reasoning-LLM phenotype extraction against the CardiffCOGS gold standard.

Pipeline: extract binary prediction -> merge on Study_ID -> compute metrics.

Usage:
    python evaluate_llm.py --prompt clozapine1
    python evaluate_llm.py --prompt clozapine1 clozapine2
    python evaluate_llm.py --prompt comp1 --gold-col Some_TRS_Variable
    python evaluate_llm.py --prompt clozapine1 --steps extract
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import pyreadstat
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
)

# ----------------------------------------------------------------------
# Configuration. Change paths and gold-standard mappings here.
# ----------------------------------------------------------------------
PROJECT_DIR = Path("/scratch/SCWF00151/c.c25097054")
DATA_DIR    = PROJECT_DIR / "treatment-response"
RESULTS_DIR = PROJECT_DIR / "results"
GOLD_SAV    = Path("/scratch/SCWF00105/sarah_msc/Complete CardiffCogs 2022.sav")

ID_COL   = "Study_ID"
RAW_COL  = "Generated_text"
PRED_COL = "Extract"
RAW_FILE = "cogs_predict_tr.csv"
THINK_TAG = "</think>"

# Gold-standard variable for each prompt.
# None means the correct gold standard is not yet confirmed, so the script
# will refuse to score it rather than compare against the wrong variable.
GOLD_COLUMN = {
    "clozapine1": "Clozapine_Ever",
    "clozapine2": "Clozapine_Ever",
    "comp1":    None,
    "comp2":    None,
    "failure1": None,
    "failure2": None,
    "failure3": None,
    "adr1":     None,
    "adr2":     None,
    "adr3":     None,
    "adr4":     None,
}


# ----------------------------------------------------------------------
# Step 1: extraction
# ----------------------------------------------------------------------

def extract_answer(text):
    """Extract the model's final binary (0/1) answer from a reasoning trace.

    Text after the </think> tag is searched for an explicit "Answer:" label
    first, falling back to a digit that appears alone on its own line. This
    avoids false matches on digits embedded in dates, dosages, or list
    numbering elsewhere in the response.

    Returns NA if no answer can be identified.
    """
    if not isinstance(text, str):
        return pd.NA
    tail = text.split(THINK_TAG)[-1] if THINK_TAG in text else text

    # Priority 1: explicit "Answer:" label
    m = re.search(r"answer:?\s*\**\s*([01])\b", tail, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Priority 2: a digit alone on its own line
    matches = re.findall(r"^\s*\**([01])\**\s*$", tail, re.MULTILINE)
    return int(matches[-1]) if matches else pd.NA

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
# ----------------------------------------------------------------------
# Step 2: merge with gold standard
# ----------------------------------------------------------------------
def merge(prompt, gold_col):
    cogs, _ = pyreadstat.read_sav(GOLD_SAV)
    gold = cogs[[ID_COL, gold_col]].copy()

    llm = pd.read_csv(RESULTS_DIR / f"{prompt}_extracted.csv")
    merged = llm.merge(gold, on=ID_COL, how="left")

    dest = RESULTS_DIR / f"{prompt}_merged.csv"
    merged.to_csv(dest, index=False)

    print(f"[{prompt}] merged on {ID_COL} against {gold_col} | "
          f"{len(merged)} rows | "
          f"matched gold: {merged[gold_col].notna().sum()} | "
          f"no gold: {merged[gold_col].isna().sum()}")
    return merged


# ----------------------------------------------------------------------
# Step 3: evaluation
# ----------------------------------------------------------------------
def evaluate(prompt, gold_col):
    """Score the LLM against the gold standard.

    Rows are dropped where either side is missing: the LLM did not commit to
    an answer, or the patient has no gold-standard label. n_scorable records
    the denominator actually used.
    """
    m = pd.read_csv(RESULTS_DIR / f"{prompt}_merged.csv")
    n_total = len(m)
    m = m.dropna(subset=[PRED_COL, gold_col])

    y_true = m[gold_col].astype(int)
    y_pred = m[PRED_COL].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "prompt": prompt,
        "gold_col": gold_col,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "n_scorable": len(m),
        "n_total": n_total,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run(prompts, gold_override=None, steps="all"):
    results = []

    for prompt in prompts:
        gold_col = gold_override or GOLD_COLUMN.get(prompt)

        if steps in ("all", "extract"):
            extract(prompt)

        if steps == "extract":
            continue

        if gold_col is None:
            print(f"[{prompt}] SKIPPED: no gold standard confirmed for this "
                  f"prompt. Set it in GOLD_COLUMN or pass --gold-col.")
            continue

        if steps in ("all", "merge"):
            merge(prompt, gold_col)

        if steps in ("all", "evaluate"):
            results.append(evaluate(prompt, gold_col))

    if not results:
        return

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

    dest = RESULTS_DIR / "evaluation_metrics.csv"
    df.to_csv(dest, index=False)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True, nargs="+",
                   help="one or more prompt names, e.g. clozapine1 comp1")
    p.add_argument("--gold-col", default=None,
                   help="override the gold-standard column from GOLD_COLUMN")
    p.add_argument("--steps", default="all",
                   choices=["all", "extract", "merge", "evaluate"],
                   help="run the whole pipeline or a single step")
    a = p.parse_args()
    run(a.prompt, a.gold_col, a.steps)
