"""
Code the ADR (general antipsychotic side-effect) gold-standard variable
from CardiffCOGS.

Restricted to participants with a schizophrenia or schizoaffective
diagnosis, on antipsychotic medication, and genotyped, matching the
sample used throughout the clinical and genetic validation pipeline.

adr_present is derived from Typical_AntiP_Side_Effects and
Atypical_AntiP_Side_Effects:
    0:  both columns are coded 0 (no documented side effect)
    1:  either column is coded 1-6, non-0 and non-88/99
        (a documented side effect)
    NA: both columns are coded 88/99/NA (side-effect status not
        applicable, not assessed, or not recorded)

Missing side-effect status is excluded rather than recoded as a
control. An alternative coding, recoding missing status as 0, was
tested and rejected: it yields a better-balanced variable (194
controls rather than 49), but treats "status unknown" as equivalent to
"confirmed no side effect", introducing non-differential
misclassification into the control group. This kind of
misclassification is expected to bias associations toward the null,
consistent with what was observed: the TRS PGS association with the
LLM-predicted ADR outcome was attenuated and non-significant under the
recoded version, and reached significance once excluded-NA cases were
correctly treated as missing rather than as controls. Excluding
missing status is therefore retained as the primary coding, despite
the resulting small control group (n=49) limiting statistical power in
downstream models.
"""
import pandas as pd
import pyreadstat
from pathlib import Path

GOLD_SAV = "/scratch/SCWF00105/sarah_msc/Complete CardiffCogs 2022.sav"
RESULTS_DIR = Path("/scratch/SCWF00151/c.c25097054/results")

cogs, meta = pyreadstat.read_sav(GOLD_SAV)

cogs = cogs[
    (cogs["DSM_diagnosis_new"].isin([1, 2, 3])) &
    (cogs["Antipsychotics"] == 1) &
    (cogs["Genotyped"] == 1)
].copy()
print(f"restricted sample size: {len(cogs)}")


def code_adr(row):
    typ = row["Typical_AntiP_Side_Effects"]
    atyp = row["Atypical_AntiP_Side_Effects"]

    typ_valid = typ not in [88, 99] and not pd.isna(typ)
    atyp_valid = atyp not in [88, 99] and not pd.isna(atyp)

    if typ == 0 and atyp == 0:
        return 0
    if (typ_valid and typ != 0) or (atyp_valid and atyp != 0):
        return 1
    return pd.NA


cogs["adr_present"] = cogs.apply(code_adr, axis=1)

out_cols = ["Study_ID", "adr_present"]
cogs[out_cols].to_csv(RESULTS_DIR / "gold_standard_adr.csv", index=False)

print("\n=== ADR class balance ===")
print(cogs["adr_present"].value_counts(dropna=False))
n_scorable = (cogs['adr_present'] == 1).sum() + (cogs['adr_present'] == 0).sum()
print(f"\nn cases: {(cogs['adr_present'] == 1).sum()}")
print(f"n controls: {(cogs['adr_present'] == 0).sum()}")
print(f"n excluded (NA): {cogs['adr_present'].isna().sum()}")
if n_scorable > 0:
    pct_case = (cogs['adr_present'] == 1).sum() / n_scorable * 100
    print(f"% cases (of scorable): {pct_case:.1f}%")

print(f"\nsaved {RESULTS_DIR / 'gold_standard_adr.csv'}")
