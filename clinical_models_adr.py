"""
Clinical validation regressions for ADR.

For each clinical predictor, fit a logistic regression twice:
    Model A: outcome = adr_present   (gold standard)
    Model B: outcome = LLM_ADR       (LLM prediction)

Both models in a pair are fitted on the same complete-case sample, so
any difference in effect size is attributable to the outcome variable
rather than to differing samples.

Raw p-values are corrected for multiple comparisons using the
Benjamini-Hochberg false discovery rate procedure (Benjamini and
Hochberg, 1995), applied separately within each outcome (gold and LLM
corrected independently, each across the 5 clinical predictors).

Usage:
    python clinical_models_adr.py
    python clinical_models_adr.py --predictor NARTIQ
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from pathlib import Path

RESULTS_DIR = Path("/scratch/SCWF00151/c.c25097054/results")

GOLD_OUTCOME = "adr_present"
LLM_OUTCOME = "LLM_ADR"

PREDICTORS = [
    "OPCRIT_age_of_onset_psychosis",
    "NARTIQ",
    "OPCRIT_DandH_9_poor_premorbid_work_adjustment",
    "OPCRIT_DandH_10_poor_premorbid_social_adjustment",
    "Derived_GAS_psychosis",
]

COVARIATES = ["Age_at_Interview", "male_sex", "C(DSM_diagnosis_new)", "Adherence"]


def fit_pair(df, predictor):
    needed = [predictor, GOLD_OUTCOME, LLM_OUTCOME,
              "Age_at_Interview", "male_sex", "DSM_diagnosis_new", "Adherence"]
    sample = df[needed].dropna()

    rhs = " + ".join([predictor] + COVARIATES)
    out = []

    for label, outcome in [("gold", GOLD_OUTCOME), ("LLM", LLM_OUTCOME)]:
        model = smf.glm(f"{outcome} ~ {rhs}", data=sample,
                        family=sm.families.Binomial()).fit()
        beta = model.params[predictor]
        ci_low, ci_high = model.conf_int().loc[predictor]
        out.append({
            "predictor": predictor,
            "outcome": label,
            "n": int(model.nobs),
            "beta": beta,
            "se": model.bse[predictor],
            "OR": np.exp(beta),
            "OR_low": np.exp(ci_low),
            "OR_high": np.exp(ci_high),
            "z": model.tvalues[predictor],
            "p": model.pvalues[predictor],
        })
    return out


def main(predictors):
    df = pd.read_csv(RESULTS_DIR / "analysis_adr.csv", low_memory=False)

    rows = []
    for p in predictors:
        if p not in df.columns:
            print(f"SKIPPED: {p} not in dataset")
            continue
        rows.extend(fit_pair(df, p))

    res = pd.DataFrame(rows)

    res["p_fdr"] = np.nan
    for outcome in res["outcome"].unique():
        mask = res["outcome"] == outcome
        _, p_adj, _, _ = multipletests(res.loc[mask, "p"], method="fdr_bh")
        res.loc[mask, "p_fdr"] = p_adj

    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format="%.3f"))

    dest = RESULTS_DIR / "clinical_models_adr.csv"
    res.to_csv(dest, index=False)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictor", nargs="+", default=PREDICTORS)
    main(p.parse_args().predictor)
