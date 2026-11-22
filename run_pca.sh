#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Within-cohort PCA for CardiffCOGS, run for both QC levels (qc4, qc5).
# No reference panel: PCA is computed on the cohort's own pruned autosomal SNPs.
# Reads from the shared (read-only) COGS directory; writes only to user scratch.
###############################################################################

SHARED="/scratch/SCWF00105/shared/COGS"
OUTDIR="/scratch/SCWF00151/c.c25097054/genetics"
PLINK="plink"          # PLINK/1.9b_6.21 loaded via module
NPCS=20

# Long-range LD regions to exclude (build 37 / hg19), incl. MHC on chr6.
# These regions distort principal components if left in.
cat > "${OUTDIR}/highLD_b37.txt" << 'REGIONS'
1 48000000 52000000 lowLD_1
2 86000000 100500000 lowLD_2
2 134500000 138000000 lowLD_3
2 183000000 190000000 lowLD_4
3 47500000 50000000 lowLD_5
3 83500000 87000000 lowLD_6
5 44500000 50500000 lowLD_7
5 129000000 132000000 lowLD_8
6 25000000 35000000 MHC
6 57000000 64000000 lowLD_10
7 55000000 66000000 lowLD_11
8 8000000 12000000 lowLD_12
8 43000000 50000000 lowLD_13
8 112000000 115000000 lowLD_14
10 37000000 43000000 lowLD_15
11 46000000 57000000 lowLD_16
11 87500000 90500000 lowLD_17
12 33000000 40000000 lowLD_18
20 32000000 34500000 lowLD_19
REGIONS

for QC in qc4 qc5; do
  echo "=============================================================="
  echo "[INFO] Processing ${QC}"
  echo "=============================================================="

  # Build an absolute-path merge list in scratch from the shared prefixes.
  awk -v d="${SHARED}" '{print d"/"$1}' \
      "${SHARED}/filenames.${QC}.txt" > "${OUTDIR}/mergelist.${QC}.txt"

  # 1) Merge all chromosomes into one fileset
  echo "[INFO] Merging chromosomes..."
  "${PLINK}" \
    --merge-list "${OUTDIR}/mergelist.${QC}.txt" \
    --make-bed \
    --out "${OUTDIR}/COGS.${QC}.merged"

  # 2) Common-variant filter + genotyping-rate filter
  echo "[INFO] MAF and missingness filter..."
  "${PLINK}" \
    --bfile "${OUTDIR}/COGS.${QC}.merged" \
    --maf 0.05 \
    --geno 0.02 \
    --autosome \
    --make-bed \
    --out "${OUTDIR}/COGS.${QC}.common"

  # 3) LD-prune, excluding long-range LD regions
  echo "[INFO] LD pruning..."
  "${PLINK}" \
    --bfile "${OUTDIR}/COGS.${QC}.common" \
    --exclude range "${OUTDIR}/highLD_b37.txt" \
    --indep-pairwise 50 5 0.2 \
    --out "${OUTDIR}/COGS.${QC}.prune"

  "${PLINK}" \
    --bfile "${OUTDIR}/COGS.${QC}.common" \
    --extract "${OUTDIR}/COGS.${QC}.prune.prune.in" \
    --make-bed \
    --out "${OUTDIR}/COGS.${QC}.pruned"

  # 4) PCA
  echo "[INFO] Running PCA (${NPCS} PCs)..."
  "${PLINK}" \
    --bfile "${OUTDIR}/COGS.${QC}.pruned" \
    --pca "${NPCS}" \
    --out "${OUTDIR}/COGS.${QC}.pca"

  echo "[DONE] ${QC}: ${OUTDIR}/COGS.${QC}.pca.eigenvec / .eigenval"
done
