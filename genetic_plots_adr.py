"""
Genetic validation plots for ADR.

Produces, for TRS PGS and SCZ PGS: a predicted probability curve, an
OR forest plot comparing the gold-standard and LLM-predicted outcomes,
and a PGS distribution plot by ADR status. Also produces a forest plot
for the combined model.

Uses the same complete-case sample as genetic_models_adr.py: TRS PGS,
SCZ PGS, PC1-5, and total antipsychotic count (total_antipsych_1) are
all required to be present.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/scratch/SCWF00151/c.c25097054/results"
COVARIATE_FILE = "/scratch/SCWF00151/c.c25097054/lifetime_antipsych_Sarah.txt"

GOLD, LLM = "adr_present", "LLM_ADR"
PCS = [f"PC{i}" for i in range(1, 6)]
COVARS = ["Age_at_Interview", "male_sex", "C(DSM_diagnosis_new)",
          "Adherence", "total_antipsych_1"]
colors = {"gold": "#c0392b", "LLM": "#2980b9"}

df = pd.read_csv(f"{RESULTS}/analysis_adr_genetic.csv", low_memory=False)
df["Study_ID"] = df["Study_ID"].astype(str)

antipsych = pd.read_csv(COVARIATE_FILE, sep=r"\s+")
antipsych["Study_ID"] = antipsych["Study_ID"].astype(str)
df = df.merge(antipsych, on="Study_ID", how="left")

df["TRS_PRS_z"] = (df["TRS_PRS"] - df["TRS_PRS"].mean()) / df["TRS_PRS"].std()
df["SCZ_PRS_z"] = (df["SCZ_PRS"] - df["SCZ_PRS"].mean()) / df["SCZ_PRS"].std()


def three_panel(predictor_z, other_predictor_z, pgs_label, results_csv, out_png):
    rhs = " + ".join([predictor_z] + PCS + COVARS)
    needed = [predictor_z, other_predictor_z, GOLD, LLM] + PCS + \
             ["Age_at_Interview", "male_sex", "DSM_diagnosis_new",
              "Adherence", "total_antipsych_1"]
    sample = df[list(dict.fromkeys(needed))].dropna().copy()

    models = {}
    for label, outcome in [("gold", GOLD), ("LLM", LLM)]:
        models[label] = smf.glm(f"{outcome} ~ {rhs}", data=sample,
                                family=sm.families.Binomial()).fit()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    grid_x = np.linspace(sample[predictor_z].quantile(0.02),
                         sample[predictor_z].quantile(0.98), 60)
    base = {"Age_at_Interview": sample["Age_at_Interview"].mean(),
            "male_sex": sample["male_sex"].mode()[0],
            "DSM_diagnosis_new": sample["DSM_diagnosis_new"].mode()[0],
            "Adherence": sample["Adherence"].mean(),
            "total_antipsych_1": sample["total_antipsych_1"].mean()}
    for pc in PCS:
        base[pc] = sample[pc].mean()
    for label in ["gold", "LLM"]:
        grid = pd.DataFrame({**{k: [v]*len(grid_x) for k, v in base.items()},
                             predictor_z: grid_x})
        pred = models[label].get_prediction(grid).summary_frame()
        ax.plot(grid_x, pred["mean"], color=colors[label], label=label, lw=2)
        ax.fill_between(grid_x, pred["mean_ci_lower"], pred["mean_ci_upper"],
                        color=colors[label], alpha=0.15)
    ax.set_xlabel(f"{pgs_label} (SD)"); ax.set_ylabel("P(ADR)")
    ax.set_title(f"Predicted probability vs {pgs_label}"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    res = pd.read_csv(f"{RESULTS}/{results_csv}")
    for i, (_, r) in enumerate(res.iterrows()):
        yp = 1 - i
        ax.errorbar(r["OR"], yp,
                    xerr=[[r["OR"]-r["OR_low"]], [r["OR_high"]-r["OR"]]],
                    fmt="o", ms=10, capsize=5, color=colors[r["outcome"]])
    ax.axvline(1.0, color="grey", ls="--", alpha=0.7)
    ax.set_yticks([1, 0]); ax.set_yticklabels(res["outcome"]); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel(f"OR per SD {pgs_label} (95% CI)")
    ax.set_title(f"{pgs_label} effect: gold vs LLM"); ax.grid(alpha=0.3, axis="x")

    ax = axes[2]
    for val, name, col in [(0, "no", "#7f8c8d"), (1, "yes", "#c0392b")]:
        sub = sample[sample[GOLD] == val][predictor_z]
        ax.hist(sub, bins=15, alpha=0.5, density=True, color=col,
                label=f"ADR {name} (n={len(sub)})")
    ax.set_xlabel(f"{pgs_label} (SD)"); ax.set_ylabel("density")
    ax.set_title(f"{pgs_label} by ADR status"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/{out_png}", dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


def combined_forest(out_png):
    res = pd.read_csv(f"{RESULTS}/genetic_models_adr_combined.csv")
    pretty_names = {"SCZ_PRS_z": "SCZ PGS", "TRS_PRS_z": "TRS PGS"}
    order = [
        ("TRS_PRS_z", "gold"), ("TRS_PRS_z", "LLM"),
        ("SCZ_PRS_z", "gold"), ("SCZ_PRS_z", "LLM"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    for i, (predictor, outcome) in enumerate(order):
        yp = len(order) - 1 - i
        r = res[(res["predictor"] == predictor) & (res["outcome"] == outcome)].iloc[0]
        ax.errorbar(r["OR"], yp,
                    xerr=[[r["OR"] - r["OR_low"]], [r["OR_high"] - r["OR"]]],
                    fmt="o", ms=10, capsize=5, color=colors[outcome])
        labels.append(f"{pretty_names[predictor]} ({outcome})")

    ax.axvline(1.0, color="grey", ls="--", alpha=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("OR per SD (95% CI)")
    ax.set_title("Combined model: SCZ PGS and TRS PGS effects (ADR, gold vs LLM)")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/{out_png}", dpi=150)
    plt.close(fig)
    print(f"saved {out_png}")


three_panel("TRS_PRS_z", "SCZ_PRS_z", "TRS PGS",
            "genetic_models_adr_trs.csv", "genetic_regression_plots_adr_trs.png")
three_panel("SCZ_PRS_z", "TRS_PRS_z", "SCZ PGS",
            "genetic_models_adr_scz.csv", "genetic_regression_plots_adr_scz.png")
combined_forest("genetic_regression_plots_adr_combined.png")
