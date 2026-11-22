"""Merge TRS PGS, SCZ PGS, and ancestry PCs into the ADR analysis
dataset, mirroring build_genetic_data.py (clozapine1).

Uses the same ID-matching fix (splitting the doubled sscore IID on "_")
applied throughout the genetic pipeline.
"""
import pandas as pd

GENETICS = "/scratch/SCWF00151/c.c25097054/genetics"
ANC = "/scratch/SCWF00151/c.c25097054/genetics/ancestry"
PRSCS_RESULTS = "/scratch/SCWF00151/c.c25097054/prscs/results"
RESULTS = "/scratch/SCWF00151/c.c25097054/results"

df = pd.read_csv(f"{RESULTS}/analysis_adr.csv", low_memory=False)
df["Study_ID"] = df["Study_ID"].astype(str)

stale = ["TRS_PRS", "SCZ_PRS"] + [f"PC{i}" for i in range(1, 21)]
df = df.drop(columns=[c for c in stale if c in df.columns])
print("analysis rows:", len(df))

# --- TRS polygenic score ---
prs = pd.read_csv(f"{GENETICS}/prcs_trs_score.sscore", sep=r"\s+")
prs["Study_ID"] = prs["IID"].astype(str).str.split("_").str[0]
prs = prs[["Study_ID", "SCORE1_AVG"]].rename(columns={"SCORE1_AVG": "TRS_PRS"})
df = df.merge(prs, on="Study_ID", how="left")
print("patients with TRS_PRS:", df["TRS_PRS"].notna().sum())

# --- SCZ polygenic score ---
scz = pd.read_csv(f"{PRSCS_RESULTS}/scz_score.sscore", sep=r"\s+")
scz["Study_ID"] = scz["IID"].astype(str).str.split("_").str[0]
scz = scz[["Study_ID", "SCORE1_AVG"]].rename(columns={"SCORE1_AVG": "SCZ_PRS"})
df = df.merge(scz, on="Study_ID", how="left")
print("patients with SCZ_PRS:", df["SCZ_PRS"].notna().sum())

# --- ancestry PCs ---
cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 21)]
pca = pd.read_csv(f"{ANC}/combined_pca.eigenvec", sep=r"\s+", header=None, names=cols)
pca["Study_ID"] = pca["IID"].astype(str).str.split("_").str[0]
pca = pca[["Study_ID"] + [f"PC{i}" for i in range(1, 11)]]
df = df.merge(pca, on="Study_ID", how="left")
print("patients with PCs:", df["PC1"].notna().sum())

all_three = df[df["TRS_PRS"].notna() & df["SCZ_PRS"].notna() & df["PC1"].notna()]
print("patients with TRS_PRS, SCZ_PRS, and PCs:", len(all_three))

df.to_csv(f"{RESULTS}/analysis_adr_genetic.csv", index=False)
print("saved analysis_adr_genetic.csv")
