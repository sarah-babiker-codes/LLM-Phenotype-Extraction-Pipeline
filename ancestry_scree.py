"""
Scree plot for the reference-merged ancestry PCA.

Reads the per-component eigenvalues from PLINK2's PCA output
(combined_pca.eigenval, computed on the pruned, high-LD-excluded
genotype set merged with the 1000 Genomes Phase 3 reference panel;
see run_pca.sh) and plots the percentage of variance each principal
component explains. Used to justify how many PCs to retain as
covariates in clinical and genetic validation: the elbow in this plot
(a sharp drop in variance explained after PC4/PC5) is the basis for
retaining the first 5 PCs throughout this research.

Usage:
    python ancestry_scree.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANC = "/scratch/SCWF00151/c.c25097054/genetics/ancestry"
ev = pd.read_csv(f"{ANC}/combined_pca.eigenval", header=None)[0]
pct = 100 * ev / ev.sum()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(pct) + 1), pct, "o-", color="#c0392b")
ax.set_xlabel("Principal component")
ax.set_ylabel("% variance explained")
ax.set_title("Scree plot: reference-merged ancestry PCA")
ax.set_xticks(range(1, len(pct) + 1))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ANC}/ancestry_scree.png", dpi=150)
print(f"saved {ANC}/ancestry_scree.png")
for i, p in enumerate(pct, 1):
    print(f"PC{i:2d}: {p:.2f}%")
