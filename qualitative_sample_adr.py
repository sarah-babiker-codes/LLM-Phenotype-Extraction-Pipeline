"""
Sample misclassified ADR (adr2) cases for qualitative error analysis.

Selects all false positives (n=2) plus a random sample of false
negatives, for a total sample of 20, since false positives are too few
to sample from. Outputs the clinical vignette, the LLM's full
reasoning trace, and each patient's TRS and SCZ polygenic scores (raw
and standardised) for manual review, alongside blank columns for
recording the location of the first error and its error category.

Random seed fixed for reproducibility.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path("/scratch/SCWF00151/c.c25097054")
RESULTS_DIR = PROJECT_DIR / "results"
ID_COL = "Study_ID"
SEED = 42
N_TOTAL = 20

raw = pd.read_csv(PROJECT_DIR / "treatment-response" / "dsq_adr2" / "cogs_predict_tr.csv")

def extract_note(prompt):
    note = prompt.split("Patient note:")[-1]
    note = note.split("<think>")[0]
    return note.strip()

raw["clinical_vignette"] = raw["Prompt"].apply(extract_note)
raw["Study_ID"] = raw["Study_ID"].astype(str)
raw = raw[[ID_COL, "clinical_vignette", "Generated_text"]]

merged = pd.read_csv(RESULTS_DIR / "analysis_adr.csv", low_memory=False)
merged["Study_ID"] = merged["Study_ID"].astype(str)
merged = merged.merge(raw, on=ID_COL, how="left")

genetic = pd.read_csv(RESULTS_DIR / "analysis_adr_genetic.csv", low_memory=False)
genetic["Study_ID"] = genetic["Study_ID"].astype(str)
genetic["TRS_PRS_z"] = (genetic["TRS_PRS"] - genetic["TRS_PRS"].mean()) / genetic["TRS_PRS"].std()
genetic["SCZ_PRS_z"] = (genetic["SCZ_PRS"] - genetic["SCZ_PRS"].mean()) / genetic["SCZ_PRS"].std()
merged = merged.merge(
    genetic[["Study_ID", "TRS_PRS", "TRS_PRS_z", "SCZ_PRS", "SCZ_PRS_z"]],
    on="Study_ID", how="left"
)

GOLD, LLM = "adr_present", "LLM_ADR"
merged = merged.dropna(subset=[GOLD, LLM])

fp = merged[(merged[GOLD] == 0) & (merged[LLM] == 1)].copy()
fn = merged[(merged[GOLD] == 1) & (merged[LLM] == 0)].copy()
fp["error_type"] = "FP"
fn["error_type"] = "FN"

print(f"total FP: {len(fp)}  total FN: {len(fn)}")

n_fn_needed = N_TOTAL - len(fp)
fn_sample = fn.sample(n=n_fn_needed, random_state=SEED)

sample = pd.concat([fp, fn_sample], ignore_index=True)
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

out_cols = [ID_COL, "error_type", "clinical_vignette", "Generated_text",
            "TRS_PRS", "TRS_PRS_z", "SCZ_PRS", "SCZ_PRS_z"]
out = sample[out_cols].copy()
out["first_error_location"] = ""
out["downstream_errors"] = ""
out["error_category"] = ""

dest = RESULTS_DIR / "qualitative_sample_adr.csv"
out.to_csv(dest, index=False)
print(f"sampled {len(out)} cases ({(out['error_type'] == 'FP').sum()} FP, "
      f"{(out['error_type'] == 'FN').sum()} FN)")
n_with_pgs = out["TRS_PRS"].notna().sum()
print(f"cases with PGS data: {n_with_pgs} of {len(out)}")
print(f"saved {dest}")
