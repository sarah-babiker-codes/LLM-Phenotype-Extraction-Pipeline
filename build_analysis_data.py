"""Join LLM predictions to the full CardiffCOGS record for validation modelling."""
import pandas as pd
import pyreadstat
from pathlib import Path

RESULTS_DIR = Path("/scratch/SCWF00151/c.c25097054/results")
GOLD_SAV = "/scratch/SCWF00105/sarah_msc/Complete CardiffCogs 2022.sav"
ID_COL = "Study_ID"

cogs, meta = pyreadstat.read_sav(GOLD_SAV)
llm = pd.read_csv(RESULTS_DIR / "clozapine1_extracted.csv")
llm = llm.rename(columns={"Extract": "LLM_Clozapine"})

df = llm.merge(cogs, on=ID_COL, how="left")
df.to_csv(RESULTS_DIR / "analysis_clozapine1.csv", index=False)
print(f"analysis dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# missingness within the analysis sample, not the full cohort
model_vars = {
    "outcome (gold)":    "Clozapine_Ever",
    "outcome (LLM)":     "LLM_Clozapine",
    "age of onset (A)":  "OPCRIT_age_of_onset_psychosis",
    "age of onset (B)":  "Psychosis_AAO",
    "NARTIQ":            "NARTIQ",
    "premorbid work":    "OPCRIT_DandH_9_poor_premorbid_work_adjustment",
    "premorbid social":  "OPCRIT_DandH_10_poor_premorbid_social_adjustment",
    "GAS derived psych": "Derived_GAS_psychosis",
    "GAS past week":     "GAS_Past_Week",
    "adherence":         "Adherence",
    "age":               "Age_at_Interview",
    "sex":               "male_sex",
    "diagnosis":         "DSM_diagnosis_new",
}

print(f"\nmissingness within the {len(df)} LLM patients:")
for label, col in model_vars.items():
    if col in df.columns:
        n_miss = df[col].isna().sum()
        print(f"  {label:20s} {col:50s} missing {n_miss:4d} "
              f"({100*n_miss/len(df):.1f}%)")
    else:
        print(f"  {label:20s} {col:50s} NOT FOUND")

# do the two age-of-onset variables agree where both are present?
a, b = "OPCRIT_age_of_onset_psychosis", "Psychosis_AAO"
if a in df.columns and b in df.columns:
    both = df[[a, b]].dropna()
    print(f"\nage-of-onset candidates: {len(both)} patients have both")
    if len(both):
        print(f"  correlation: {both[a].corr(both[b]):.3f}")
        print(f"  identical:   {(both[a] == both[b]).sum()} of {len(both)}")

# complete-case sample size for the first model
core = ["OPCRIT_age_of_onset_psychosis", "Age_at_Interview", "male_sex",
        "DSM_diagnosis_new", "Adherence", "Clozapine_Ever", "LLM_Clozapine"]
core = [c for c in core if c in df.columns]
print(f"\ncomplete cases for age-of-onset model: {df[core].dropna().shape[0]} of {len(df)}")
