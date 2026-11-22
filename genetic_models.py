"""
Genetic validation regressions for clozapine1: TRS PGS univariate,
SCZ PGS univariate, and the combined model with both PGS together.

The complete-case sample is standardised across all three models
(TRS_PRS, SCZ_PRS, PCs, and all covariates required regardless of
which PGS is the model's predictor), so effect sizes are directly
comparable across models. Principal components are limited to the
first 5, matching the scree plot elbow.

Raw p-values from the two univariate models (TRS PGS alone, SCZ PGS
alone) are corrected for multiple comparisons using the
Benjamini-Hochberg false discovery rate procedure (Benjamini and
Hochberg, 1995), applied separately within each outcome (gold and LLM
corrected independently, each across the 2 univariate p-values). The
combined model, in which both PGS are fitted as predictors in the same
regression, is not corrected, since the model itself jointly accounts
for both predictors rather than testing them as separate hypotheses.

Usage:
    python genetic_models.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

RESULTS = Path("/scratch/SCWF00151/c.c25097054/results")

GOLD, LLM = "Clozapine_Ever", "LLM_Clozapine"
PCS = [f"PC{i}" for i in range(1, 6)]
COVARS = ["Age_at_Interview", "male_sex", "C(DSM_diagnosis_new)", "Adherence"]

df = pd.read_csv(RESULTS / "analysis_genetic.csv", low_memory=False)
df["TRS_PRS_z"] = (df["TRS_PRS"] - df["TRS_PRS"].mean()) / df["TRS_PRS"].std()
df["SCZ_PRS_z"] = (df["SCZ_PRS"] - df["SCZ_PRS"].mean()) / df["SCZ_PRS"].std()


def fit_univariate(predictor_z, other_predictor_z):
    rhs = " + ".join([predictor_z] + PCS + COVARS)
    needed = [predictor_z, other_predictor_z, GOLD, LLM] + PCS + \
             ["Age_at_Interview", "male_sex", "DSM_diagnosis_new", "Adherence"]
    sample = df[list(dict.fromkeys(needed))].dropna()

    rows = []
    for label, outcome in [("gold", GOLD), ("LLM", LLM)]:
        m = smf.glm(f"{outcome} ~ {rhs}", data=sample,
                    family=sm.families.Binomial()).fit()
        beta = m.params[predictor_z]
        lo, hi = m.conf_int().loc[predictor_z]
        rows.append({
            "predictor": predictor_z, "outcome": label, "n": int(m.nobs),
            "beta": beta, "se": m.bse[predictor_z],
            "OR": np.exp(beta), "OR_low": np.exp(lo), "OR_high": np.exp(hi),
            "z": m.tvalues[predictor_z],
            "p": m.pvalues[predictor_z],
        })
    return rows, len(sample)


def fit_combined():
    predictors = ["SCZ_PRS_z", "TRS_PRS_z"]
    rhs = " + ".join(predictors + PCS + COVARS)
    needed = predictors + [GOLD, LLM, "Age_at_Interview", "male_sex",
                            "DSM_diagnosis_new", "Adherence"] + PCS
    sample = df[needed].dropna()

    rows = []
    for label, outcome in [("gold", GOLD), ("LLM", LLM)]:
        m = smf.glm(f"{outcome} ~ {rhs}", data=sample,
                    family=sm.families.Binomial()).fit()
        for predictor in predictors:
            beta = m.params[predictor]
            lo, hi = m.conf_int().loc[predictor]
            rows.append({
                "outcome": label, "predictor": predictor, "n": int(m.nobs),
                "beta": beta, "se": m.bse[predictor],
                "OR": np.exp(beta), "OR_low": np.exp(lo), "OR_high": np.exp(hi),
                "z": m.tvalues[predictor],
                "p": m.pvalues[predictor],
            })
    return rows, len(sample)


pd.set_option("display.width", 200)

trs_rows, n_trs = fit_univariate("TRS_PRS_z", "SCZ_PRS_z")
trs_res = pd.DataFrame(trs_rows)

scz_rows, n_scz = fit_univariate("SCZ_PRS_z", "TRS_PRS_z")
scz_res = pd.DataFrame(scz_rows)

uni_res = pd.concat([trs_res, scz_res], ignore_index=True)
uni_res["p_fdr"] = np.nan
for outcome in uni_res["outcome"].unique():
    mask = uni_res["outcome"] == outcome
    _, p_adj, _, _ = multipletests(uni_res.loc[mask, "p"], method="fdr_bh")
    uni_res.loc[mask, "p_fdr"] = p_adj

trs_res_final = uni_res[uni_res["predictor"] == "TRS_PRS_z"].reset_index(drop=True)
scz_res_final = uni_res[uni_res["predictor"] == "SCZ_PRS_z"].reset_index(drop=True)

print(f"\n=== TRS PGS univariate (n={n_trs}) ===")
print(trs_res_final.to_string(index=False, float_format="%.4f"))
trs_res_final.to_csv(RESULTS / "genetic_models_trs.csv", index=False)

print(f"\n=== SCZ PGS univariate (n={n_scz}) ===")
print(scz_res_final.to_string(index=False, float_format="%.4f"))
scz_res_final.to_csv(RESULTS / "genetic_models_scz.csv", index=False)

comb_rows, n_comb = fit_combined()
comb_res = pd.DataFrame(comb_rows)
print(f"\n=== combined SCZ + TRS PGS (n={n_comb}), not FDR-corrected ===")
print(comb_res.to_string(index=False, float_format="%.4f"))
comb_res.to_csv(RESULTS / "genetic_models_combined.csv", index=False)

print(f"\nsaved genetic_models_trs.csv, genetic_models_scz.csv, "
      f"genetic_models_combined.csv")
