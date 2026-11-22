"""Join the LLM's ADR prediction and the derived ADR gold standard to
the full CardiffCOGS record, for clinical and genetic validation.

adr_present is a derived variable (see code_gold_standard_adr.py), not
a raw CardiffCOGS column, so it is merged in from gold_standard_adr.csv
rather than read from the .sav file directly.

LLM predictions use adr2: of the two general-ADR prompt phrasings
evaluated (adr1, adr2), adr2 achieves the higher macro F1 with complete
answer coverage across all 704 patients (see evaluate_adr.py).
"""
import pandas as pd
import pyreadstat
from pathlib import Path

RESULTS_DIR = Path("/scratch/SCWF00151/c.c25097054/results")
GOLD_SAV = "/scratch/SCWF00105/sarah_msc/Complete CardiffCogs 2022.sav"
ID_COL = "Study_ID"

cogs, meta = pyreadstat.read_sav(GOLD_SAV)
gold_adr = pd.read_csv(RESULTS_DIR / "gold_standard_adr.csv")

llm = pd.read_csv(RESULTS_DIR / "adr2_extracted.csv")
llm = llm.rename(columns={"Extract": "LLM_ADR"})

df = llm.merge(cogs, on=ID_COL, how="left")
df = df.merge(gold_adr[[ID_COL, "adr_present"]], on=ID_COL, how="left")

df.to_csv(RESULTS_DIR / "analysis_adr.csv", index=False)
print(f"analysis dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"adr_present missing: {df['adr_present'].isna().sum()}")
print(f"LLM_ADR missing: {df['LLM_ADR'].isna().sum()}")
