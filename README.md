# MSc Dissertation: LLM Phenotype Extraction Pipeline

Evaluation of a reasoning LLM (DeepSeek-R1-Distill-Qwen-32B) for extracting
clinical phenotypes from CardiffCOGS clinical vignette notes, validated against clinical
and genetic gold standards.

Two phenotypes are covered: clozapine use (a proxy for treatment-resistant
schizophrenia) and ADR (general antipsychotic side effects).

## Pipeline order

### 0. Cohort description

- `descriptive_statistics.py` — reports sample size, sex, age, and
  ethnicity for the full cohort, the restricted ADR gold-standard
  sample, and the LLM-run sample

Each stage depends on the output of the previous one. Scripts read from and
write to `results/`, which is not version-controlled (see Data below);
outputs must be regenerated locally by running the scripts in order.

### 1. LLM evaluation

- `evaluate_llm.py` — extracts, merges, and scores clozapine1/clozapine2
  predictions against `Clozapine_Ever`
- `evaluate_adr.py` — extracts, merges, and scores adr1/adr2 predictions
  against `adr_present` (see step 2 for the gold standard)

### 2. Gold standard coding

- `code_gold_standard_adr.py` — codes `adr_present` from CardiffCOGS,
  restricted to schizophrenia/schizoaffective diagnosis, on antipsychotic
  medication, genotyped. Run before `evaluate_adr.py`.

### 3. Analysis dataset construction

- `build_analysis_data.py` — joins clozapine1 LLM predictions to the full
  CardiffCOGS record, producing `analysis_clozapine1.csv`
- `build_analysis_data_adr.py` — joins adr2 LLM predictions and
  `adr_present` to the full CardiffCOGS record, producing `analysis_adr.csv`

### 4. Clinical validation

- `clinical_models.py` / `clinical_plots.py` — clozapine1
- `clinical_models_adr.py` / `clinical_plots_adr.py` — ADR

### 5. Genetic validation

- `ancestry_scree.py`, `ancestry_plot.py`, `pc_scatter.py` — ancestry PCA,
  scree plot, and outlier QC (run once, shared by both phenotypes)
- `build_genetic_data.py` — merges TRS PGS, SCZ PGS, and PCs into
  `analysis_genetic.csv` (clozapine1)
- `build_genetic_data_adr.py` — same, producing `analysis_adr_genetic.csv`
  (ADR)
- `genetic_models.py` / `genetic_plots.py` — clozapine1: fits TRS PGS
  univariate, SCZ PGS univariate, and the combined model, and produces the
  corresponding plots, all in one script each
- `genetic_models_adr.py` / `genetic_plots_adr.py` — ADR: same structure,
  additionally including total lifetime antipsychotic count as a
  covariate

### 6. Error analysis

- `error_analysis_clozapine1.py` / `error_analysis_adr.py` — quantitative:
  tests whether note length, sex, age, or diagnosis group predict correct
  classification
- `qualitative_sample_clozapine1.py` / `qualitative_sample_adr.py` —
  samples 20 misclassified cases per phenotype for manual review

## Data

Patient data (`.csv`, `.sav`, `.txt`, `treatment-response/`, `results/`,
`genetics/`) is excluded via `.gitignore` and never uploaded to this
repository; only code is version-controlled. Scripts must be run against
the underlying CardiffCOGS data on Falcon to regenerate results.
