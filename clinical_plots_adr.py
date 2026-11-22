"""
Clinical validation plot for ADR.

Forest plot comparing all five clinical predictors (gold vs LLM),
mirroring clinical_plots.py (clozapine1).
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "/scratch/SCWF00151/c.c25097054/results"
res = pd.read_csv(f"{RESULTS}/clinical_models_adr.csv")

colors = {"gold": "#c0392b", "LLM": "#2980b9"}
pretty_names = {
    "OPCRIT_age_of_onset_psychosis": "Age of onset",
    "NARTIQ": "NARTIQ",
    "OPCRIT_DandH_9_poor_premorbid_work_adjustment": "Premorbid work adjustment",
    "OPCRIT_DandH_10_poor_premorbid_social_adjustment": "Premorbid social adjustment",
    "Derived_GAS_psychosis": "GAS (psychosis)",
}
predictor_order = list(pretty_names.keys())

order = []
for predictor in predictor_order:
    order.append((predictor, "gold"))
    order.append((predictor, "LLM"))

fig, ax = plt.subplots(figsize=(9, 7))
labels = []
for i, (predictor, outcome) in enumerate(order):
    yp = len(order) - 1 - i
    match = res[(res["predictor"] == predictor) & (res["outcome"] == outcome)]
    if match.empty:
        continue
    r = match.iloc[0]
    ax.errorbar(r["OR"], yp,
                xerr=[[r["OR"] - r["OR_low"]], [r["OR_high"] - r["OR"]]],
                fmt="o", ms=9, capsize=5, color=colors[outcome])
    labels.append(f"{pretty_names[predictor]} ({outcome})")

ax.axvline(1.0, color="grey", ls="--", alpha=0.7)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(labels[::-1])
ax.set_xlabel("OR (95% CI)")
ax.set_title("Clinical validation: gold vs LLM (ADR)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{RESULTS}/clinical_regression_plots_adr.png", dpi=150)
print("saved clinical_regression_plots_adr.png")
