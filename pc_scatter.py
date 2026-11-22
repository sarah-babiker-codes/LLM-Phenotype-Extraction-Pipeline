"""
PC scatter plots for CardiffCOGS, to visually assess population structure
and flag ancestry outliers. Companion to the scree plot.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/scratch/SCWF00151/c.c25097054/genetics"
QC = "qc4"          # qc4 and qc5 are identical, so either works
SD_THRESHOLD = 6    # flag samples beyond this many SDs on any of PC1-PC10

# eigenvec columns: FID, IID, PC1..PC20
cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, 21)]
ev = pd.read_csv(f"{OUTDIR}/COGS.{QC}.pca.eigenvec",
                 sep=r"\s+", header=None, names=cols)

# identify outliers on the top 10 PCs
pc_cols = [f"PC{i}" for i in range(1, 11)]
z = (ev[pc_cols] - ev[pc_cols].mean()) / ev[pc_cols].std()
ev["outlier"] = (z.abs() > SD_THRESHOLD).any(axis=1)
n_out = ev["outlier"].sum()

# three panels of adjacent PC pairs
pairs = [("PC1", "PC2"), ("PC2", "PC3"), ("PC3", "PC4")]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (x, y) in zip(axes, pairs):
    normal = ev[~ev["outlier"]]
    out = ev[ev["outlier"]]
    ax.scatter(normal[x], normal[y], s=12, alpha=0.5, label="samples")
    if len(out):
        ax.scatter(out[x], out[y], s=40, color="red",
                   edgecolor="black", label=f"outlier (>{SD_THRESHOLD} SD)")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    ax.grid(alpha=0.3)
    if len(out):
        ax.legend(fontsize=8)

plt.suptitle(f"CardiffCOGS PC scatter ({QC}), n={len(ev)}, "
             f"{n_out} outlier(s) beyond {SD_THRESHOLD} SD")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/pc_scatter.png", dpi=150)
print(f"saved {OUTDIR}/pc_scatter.png")

print(f"\ntotal samples: {len(ev)}")
print(f"outliers flagged (>{SD_THRESHOLD} SD on PC1-10): {n_out}")
if n_out:
    print("outlier IIDs:")
    for iid in ev.loc[ev["outlier"], "IID"]:
        print(f"  {iid}")
