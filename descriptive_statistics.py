"""
Cohort descriptive statistics.

Reports sample size, sex, age, and ethnicity for the full CardiffCOGS
cohort, the restricted sample used for the ADR gold standard
(schizophrenia/schizoaffective diagnosis, on antipsychotic medication,
genotyped), and the 704-patient sample the LLM was run on.
"""
import pandas as pd
import pyreadstat
from pathlib import Path

GOLD_SAV = "/scratch/SCWF00105/sarah_msc/Complete CardiffCogs 2022.sav"
RESULTS_DIR = Path("/scratch/SCWF00151/c.c25097054/results")

ETHNICITY_LABELS = {
    1.0: "South Wales Caucasian", 2.0: "UK/Eire Caucasian",
    3.0: "West European Caucasian", 4.0: "Other Caucasian",
    5.0: "Afro-Caribbean", 6.0: "Asian", 7.0: "Mixed race", 9.0: "Unknown",
}
CAUCASIAN_CODES = [1.0, 2.0, 3.0, 4.0]


def describe(df, label):
    print(f"\n=== {label}: n={len(df)} ===")
    print("Sex:")
    print(df["male_sex"].value_counts(dropna=False))
    print("\nAge at interview:")
    print(df["Age_at_Interview"].describe())
    if "Ethnic_Origin" in df.columns:
        print("\nEthnicity:")
        counts = df["Ethnic_Origin"].map(ETHNICITY_LABELS).value_counts(dropna=False)
        print(counts)
        n_caucasian = df["Ethnic_Origin"].isin(CAUCASIAN_CODES).sum()
        n_valid = df["Ethnic_Origin"].notna().sum()
        print(f"Caucasian (codes 1-4): {n_caucasian} of {n_valid} with ethnicity recorded "
              f"({100*n_caucasian/n_valid:.1f}%)")


cogs, meta = pyreadstat.read_sav(GOLD_SAV)
cogs["Study_ID"] = cogs["Study_ID"].astype(str)

print(f"Total patients in CardiffCOGS: {len(cogs)}")
print(f"Genotyped: {(cogs['Genotyped'] == 1).sum()} "
      f"({100*(cogs['Genotyped'] == 1).sum()/len(cogs):.1f}%)")

describe(cogs, "Full cohort")

restricted = cogs[
    (cogs["DSM_diagnosis_new"].isin([1, 2, 3])) &
    (cogs["Antipsychotics"] == 1) &
    (cogs["Genotyped"] == 1)
]
describe(restricted, "Restricted sample (diagnosis + antipsychotics + genotyped)")

llm_ids = pd.read_csv(RESULTS_DIR / "analysis_clozapine1.csv", low_memory=False)["Study_ID"].astype(str)
llm_sample = cogs[cogs["Study_ID"].isin(llm_ids)]
describe(llm_sample, "LLM-run sample")

print("\nLLM-run sample diagnosis breakdown:")
print(llm_sample["DSM_diagnosis_new"].value_counts(dropna=False))
