# Folder Layout

## Top-Level

```text
TrustSOC-DERG/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ src/
│  ├─ models/
│  ├─ supporting_analysis/
│  └─ ...
├─ scripts/
│  ├─ supporting_analysis/
│  └─ ...
├─ docs/
└─ artifacts/
```

## Artifacts

```text
artifacts/
├─ models/
├─ predictions/
├─ metrics/
├─ figures/
│  ├─ models/
│  ├─ comparison/
│  ├─ robustness/
│  └─ concepts/
├─ tables/
│  ├─ benchmark/
│  ├─ analysis/
│  └─ statistics/
├─ xai/
├─ deep_analysis/
├─ practical_experiments/
├─ reports/
│  └─ summary/
└─ logs/
```

## Conventions

- `src/`: TrustSOC core pipeline, models, calibration, preprocessing, and evaluation entrypoints.
- `src/supporting_analysis/`: XAI, deep analysis, practical experiments, temporal analysis, and paper-support utilities.
- `scripts/supporting_analysis/`: optional helper entrypoints for support-only analyses.
- `figures/models/<model_name>/`: per-model training curves, confusion matrices, calibration plots.
- `figures/comparison/`: benchmark charts used across the paper.
- `figures/robustness/`: robustness and adversarial evaluation plots.
- `figures/concepts/`: pipeline diagrams and DERG concept illustrations.
- `tables/benchmark/`: dataset statistics and calibration tables.
- `tables/analysis/`: robustness and error analysis.
- `tables/statistics/`: confidence intervals, significance tests, LaTeX-ready statistical outputs.
- `reports/summary/`: compact paper-facing summaries such as `scopus_summary.md`.
- `practical_experiments/<model_name>/`: operational triage, source-shift, workload, and case-study outputs.
