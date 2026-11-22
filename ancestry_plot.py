"""
Ancestry PCA scatter plots: CardiffCOGS against 1000 Genomes Phase 3
super-populations.

Plots PC1 vs PC2 and PC2 vs PC3 from the reference-merged PCA
(combined_pca.eigenvec; see run_pca.sh), colouring points by
1000 Genomes super-population (AFR, AMR, EAS, EUR, SAS) where a
reference label is available, and labelling all remaining points
"COGS (test)", since CardiffCOGS participants have no super-population
label in the reference file and are therefore the only samples left
unmatched after the merge. Produces Figure 1 (top panels) and reports
CardiffCOGS's mean position on PC1-PC3 relative to the EUR reference
centroid, supporting the claim that CardiffCOGS participants cluster
predominantly with the European reference population.

This script does not exclude any participants; it is purely
descriptive. Ancestry-based exclusion, if any, is handled separately
(see pc_scatter.py).

Usage:
    python ancestry_plot.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANC = "/scratch/SCWF00151/c.c25097054/genetics/ancestry"
REF = "/shared/home1/c.c25097054/hawk_c.c25097054.20260203-133904/1000reference/all-1000g-phase3-chrall-mac5-v2/all-1000g-phase3-chrall-mac5-v2"

cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 21)]
pca = pd.read_csv(f"{ANC}/combined_pca.eigenvec", sep=r"\s+", header=None, names=cols)

sp = pd.read_csv(f"{REF}.super-population", sep=r"\s+", header=None,
                 names=["FID", "IID", "SPOP"])

pca = pca.merge(sp[["IID", "SPOP"]], on="IID", how="left")
pca["group"] = pca["SPOP"].fillna("COGS (test)")

colors = {"AFR": "#2ca02c", "AMR": "#ff7f0e", "EAS": "#1f77b4",
          "EUR": "#d62728", "SAS": "#9467bd", "COGS (test)": "black"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, (x, y) in zip(axes, [("PC1", "PC2"), ("PC2", "PC3")]):
    for grp, sub in pca.groupby("group"):
        is_cogs = grp == "COGS (test)"
        ax.scatter(sub[x], sub[y], s=14 if is_cogs else 8,
                   alpha=0.8 if is_cogs else 0.4,
                   c=colors.get(grp, "grey"), label=grp,
                   edgecolor="black" if is_cogs else "none",
                   linewidth=0.3)
    ax.set_xlabel(x); ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    ax.legend(fontsize=8, markerscale=2)

plt.suptitle("CardiffCOGS projected against 1000 Genomes super-populations")
plt.tight_layout()
plt.savefig(f"{ANC}/ancestry_pca.png", dpi=150)
print(f"saved {ANC}/ancestry_pca.png")

cogs = pca[pca["group"] == "COGS (test)"]
eur = pca[pca["group"] == "EUR"]
print(f"\nCOGS samples in plot: {len(cogs)}")
print(f"EUR reference:        {len(eur)}")
print("\ngroup counts:")
print(pca["group"].value_counts())
for pc in ["PC1", "PC2", "PC3"]:
    print(f"{pc}: COGS mean {cogs[pc].mean():.4f}, EUR mean {eur[pc].mean():.4f}")
