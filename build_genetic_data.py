"""
Merge genetic data into the clozapine1 analysis dataset.

Adds TRS polygenic score, SCZ polygenic score, and the first 10
ancestry principal components to analysis_clozapine1.csv, producing
analysis_genetic.csv, the dataset genetic_models.py reads from.

PLINK2 output files (.sscore, .eigenvec) store sample identifiers in
an IID column with a trailing suffix not present in CardiffCOGS's own
Study_ID convention; this is stripped identically before each merge.
Any PRS/PC columns already present in the input file (e.g. from a
previous run) are dropped before merging, so the script can be safely
re-run without accumulating duplicate columns.

Usage:
    python build_genetic_data.py
"""
import pandas as pd

GENETICS = "/scratch/SCWF00151/c.c25097054/genetics"
ANC = "/scratch/SCWF00151/c.c25097054/genetics/ancestry"
RESULTS = "/scratch/SCWF00151/c.c25097054/results"

df = pd.read_csv(f"{RESULTS}/analysis_clozapine1.csv", low_memory=False)
df["Study_ID"] = df["Study_ID"].astype(str)

stale = ["TRS_PRS", "SCZ_PRS"] + [f"PC{i}" for i in range(1, 21)]
df = df.drop(columns=[c for c in stale if c in df.columns])
print("analysis rows:", len(df))

prs = pd.read_csv(f"{GENETICS}/prcs_trs_score.sscore", sep=r"\s+")
prs["Study_ID"] = prs["IID"].astype(str).str.split("_").str[0]
prs = prs[["Study_ID", "SCORE1_AVG"]].rename(columns={"SCORE1_AVG": "TRS_PRS"})
df = df.merge(prs, on="Study_ID", how="left")
print("patients with TRS_PRS:", df["TRS_PRS"].notna().sum())

scz = pd.read_csv(f"{GENETICS.replace('genetics', 'prscs/results')}/scz_score.sscore", sep=r"\s+")
scz["Study_ID"] = scz["IID"].astype(str).str.split("_").str[0]
scz = scz[["Study_ID", "SCORE1_AVG"]].rename(columns={"SCORE1_AVG": "SCZ_PRS"})
df = df.merge(scz, on="Study_ID", how="left")
print("patients with SCZ_PRS:", df["SCZ_PRS"].notna().sum())

cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 21)]
pca = pd.read_csv(f"{ANC}/combined_pca.eigenvec", sep=r"\s+", header=None, names=cols)
pca["Study_ID"] = pca["IID"].astype(str).str.split("_").str[0]
pca = pca[["Study_ID"] + [f"PC{i}" for i in range(1, 11)]]
df = df.merge(pca, on="Study_ID", how="left")
print("patients with PCs:", df["PC1"].notna().sum())

all_three = df[df["TRS_PRS"].notna() & df["SCZ_PRS"].notna() & df["PC1"].notna()]
print("patients with TRS_PRS, SCZ_PRS, and PCs:", len(all_three))

df.to_csv(f"{RESULTS}/analysis_genetic.csv", index=False)
print("saved analysis_genetic.csv")
