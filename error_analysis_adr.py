"""
Quantitative error analysis: ADR.

Tests whether the LLM's prediction correctness (TP/TN vs FP/FN) is
associated with case note length, sex, age, diagnosis group, or total
lifetime antipsychotic count.

    correct ~ log(note_length) + sex + age + diagnosis_group + total_antipsych_1

Note length is computed from the raw prompt text sent to the LLM (the
adr2 prompt), isolating the clinical vignette itself by taking the
text after the final "Patient note:" marker and before the model's
chat-template tags. Log-transformed since note length is right-skewed.

DSM_diagnosis_new is grouped into three categories, consistent with
error_analysis_clozapine1.py:
    Schizophrenia:     codes 1, 2, 4 (Schizophrenia, Schizoaffective
                       Depressed, Other psychotic)
    Bipolar:           codes 3, 5, 6 (Schizoaffective Bipolar,
                       Bipolar 1, Bipolar 2)
    Depression_Other:  codes 7, 8 (Depression, Other)

Raw p-values across the 5 model terms are corrected for multiple
comparisons using the Benjamini-Hochberg false discovery rate
procedure (Benjamini and Hochberg, 1995).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

PROJECT_DIR = Path("/scratch/SCWF00151/c.c25097054")
RESULTS_DIR = PROJECT_DIR / "results"
COVARIATE_FILE = PROJECT_DIR / "lifetime_antipsych_Sarah.txt"
ID_COL = "Study_ID"

DIAGNOSIS_GROUP = {
    1: "Schizophrenia", 2: "Schizophrenia", 4: "Schizophrenia",
    3: "Bipolar", 5: "Bipolar", 6: "Bipolar",
    7: "Depression_Other", 8: "Depression_Other",
}

raw = pd.read_csv(PROJECT_DIR / "treatment-response" / "dsq_adr2" / "cogs_predict_tr.csv")

def extract_note(prompt):
    note = prompt.split("Patient note:")[-1]
    note = note.split("<think>")[0]
    return note.strip()

raw["note_length"] = raw["Prompt"].apply(extract_note).str.len()
raw["log_note_length"] = np.log(raw["note_length"])
note_lengths = raw[[ID_COL, "log_note_length"]]

merged = pd.read_csv(RESULTS_DIR / "analysis_adr.csv", low_memory=False)
merged["Study_ID"] = merged["Study_ID"].astype(str)
merged = merged.merge(note_lengths, on=ID_COL, how="left")

antipsych = pd.read_csv(COVARIATE_FILE, sep=r"\s+")
antipsych["Study_ID"] = antipsych["Study_ID"].astype(str)
merged = merged.merge(antipsych, on="Study_ID", how="left")

GOLD, LLM = "adr_present", "LLM_ADR"
merged = merged.dropna(subset=[GOLD, LLM])
merged["correct"] = (merged[GOLD] == merged[LLM]).astype(int)
merged["diagnosis_group"] = merged["DSM_diagnosis_new"].map(DIAGNOSIS_GROUP)

n_correct = merged["correct"].sum()
n_total = len(merged)
print(f"ADR: {n_correct} of {n_total} correctly classified "
      f"({100*n_correct/n_total:.1f}%)")

tp = ((merged[GOLD] == 1) & (merged[LLM] == 1)).sum()
tn = ((merged[GOLD] == 0) & (merged[LLM] == 0)).sum()
fp = ((merged[GOLD] == 0) & (merged[LLM] == 1)).sum()
fn = ((merged[GOLD] == 1) & (merged[LLM] == 0)).sum()
print(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")

needed = ["correct", "log_note_length", "male_sex", "Age_at_Interview",
          "diagnosis_group", "total_antipsych_1"]
sample = merged[needed].dropna()
print(f"\ncomplete-case sample for error analysis: {len(sample)}")
print(pd.crosstab(sample["diagnosis_group"], sample["correct"]))

model = smf.glm("correct ~ log_note_length + male_sex + Age_at_Interview + "
                "C(diagnosis_group) + total_antipsych_1", data=sample,
                family=sm.families.Binomial()).fit()

print("\n" + model.summary().as_text())

rows = []
for term in model.params.index:
    if term == "Intercept":
        continue
    beta = model.params[term]
    lo, hi = model.conf_int().loc[term]
    rows.append({
        "term": term, "beta": beta, "se": model.bse[term],
        "OR": np.exp(beta), "OR_low": np.exp(lo), "OR_high": np.exp(hi),
        "z": model.tvalues[term],
        "p": model.pvalues[term],
    })
res = pd.DataFrame(rows)
_, p_adj, _, _ = multipletests(res["p"], method="fdr_bh")
res["p_fdr"] = p_adj
res.to_csv(RESULTS_DIR / "error_analysis_adr.csv", index=False)
print(f"\nsaved error_analysis_adr.csv")
