# TrustSOC-Research

## Title

TrustSOC: Trust-Calibrated Multi-Evidence Cyber Reasoning Framework for SOC

## Overview

For the public GitHub upload, heavyweight local assets such as raw/processed data, trained model weights, predictions, figures, and full analysis dumps are intentionally excluded to keep the repository lightweight. They can be regenerated locally by rerunning the pipeline.

This repository implements a local, reproducible research prototype for TrustSOC — a framework that extends SOC automation by asking not only whether a model can classify a case correctly, but also **whether it knows when it should conclude, investigate, escalate, or refuse**.

The repository is designed for:

- local preprocessing with evidence extraction and DERG construction
- TrustSOC-DERG training and evaluation
- trust calibration with adaptive threshold learning
- DERG-style multi-evidence reasoning
- adversarial robustness benchmarking (7 attack types)
- explainability and evidence attribution
- temporal evidence analysis
- statistical significance testing with bootstrap CIs
- paper-ready tables (LaTeX), figures, and case studies

## Research Question

How can SOC reasoning know when to trust itself and when to refuse a conclusion under adversarial evidence conditions?

## Contributions

1. A DERG-based multi-evidence representation for SOC alerts with graph-derived features.
2. A Trust Calibration Transformer (TCT) with adaptive threshold learning for deciding when to conclude, investigate, escalate, or refuse.
3. A comprehensive adversarial SOC hallucination benchmark with 7 attack types (noise injection, evidence poisoning, evidence suppression, label manipulation, missing CTI, missing MITRE, adversarial cases).
4. A Human-AI Trust Alignment Metric for quantifying whether the system knows when to defer, escalate, or refuse.

XAI, temporal analysis, practical experiments, and statistical testing are treated as supporting analyses rather than standalone paper contributions.

## Dataset

The code uses local files in `data/raw/` from:

- TrustSOC adversarial dataset (JSONL splits)
- CTI sources (OTX, CVE, malicious domains/IPs)
- Microsoft GUIDE dataset (optional)

## Methodology

The pipeline performs:

1. Raw data copy and schema normalization
2. Evidence extraction with threat pattern matching
3. CTI and MITRE ATT&CK matching
4. DERG construction with graph-derived features
5. Model training for TrustSOC-DERG
6. Trust calibration with adaptive threshold learning
7. Robustness evaluation across 7 adversarial attack types
8. Statistical significance testing
9. Figure, table, and case study generation

## Architecture

- `TrustSOC-DERG`: PyTorch text encoder + DERG numeric features + multi-task heads + trust calibration

## DERG Construction

Each case includes alert, incident, entity, account, device, IP, domain, file, CTI, MITRE, and evidence nodes with reliability and contradiction attributes. Graph features (density, centrality, node/edge counts, etc.) are extracted for model input.

## Trust Calibration

The trust layer predicts:

- trust score (calibrated probability of correct prediction)
- uncertainty score
- reliability score
- expected action: conclude, investigate, escalate, refuse

Enhanced with:
- Adaptive threshold learning (replaces fixed thresholds)
- Temperature scaling for post-hoc calibration
- Trust score decomposition for explainability

## Adversarial SOC Hallucination Benchmark

The benchmark evaluates robustness against 7 attack types:

| Attack Type | Description |
|---|---|
| Noise Injection | Adds benign noise text |
| Evidence Poisoning | Injects fake high-reliability CTI |
| Evidence Suppression | Removes all structured evidence |
| Label Manipulation | Contradicts evidence labels |
| Missing CTI | Zeros CTI features |
| Missing MITRE | Zeros MITRE features |
| Adversarial Cases | Natural adversarial samples |

## Installation

```powershell
cd <project_directory>
python -m pip install -r requirements.txt
```

For the Transformer model, also install PyTorch:
```powershell
python -m pip install torch
```

## How to Run

```powershell
python main.py --mode preprocess
python main.py --mode train_baselines
python main.py --mode train_derg
python main.py --mode evaluate
python main.py --mode compare_opensoc
python main.py --mode robustness
python main.py --mode report
python main.py --mode xai
python main.py --mode deep_analysis
python main.py --mode practical_experiments
python main.py --mode full_analysis
```

Or use individual scripts:

```powershell
python scripts/run_preprocess.py
python scripts/train_derg.py
python scripts/evaluate.py
python scripts/compare_with_opensoc.py
python scripts/run_robustness.py
python scripts/generate_report_tables.py
python scripts/supporting_analysis/run_xai.py
python scripts/supporting_analysis/run_deep_analysis.py
python scripts/supporting_analysis/run_practical_experiments.py
python scripts/run_full_analysis.py
```

## Configuration

All paths are auto-detected relative to the project root. To override:

```powershell
# Set custom dataset location
$env:TRUSTSOC_DATASET_ROOT = "C:\path\to\your\dataset"
$env:TRUSTSOC_SOURCE_ROOT = "C:\path\to\trustsoc\data"
python main.py --mode preprocess
```

## Results

Results are generated locally and stored in:

- `artifacts/models/` — trained model bundles
- `artifacts/metrics/` — JSON metrics with bootstrap CIs
- `artifacts/predictions/` — per-sample predictions
- `artifacts/figures/` — publication-quality figures
- `artifacts/tables/` — CSV and LaTeX tables
- `artifacts/reports/` — case studies and summaries

- `artifacts/xai/` â€” SHAP, attention rollout, integrated gradients, LIME, and probing artifacts
- `artifacts/deep_analysis/` â€” subgroup calibration, MC-dropout, counterfactual, error decomposition, and behavioral testing artifacts

- `artifacts/figures/models/` â€” per-model training and evaluation figures
- `artifacts/figures/comparison/` â€” benchmark and paper comparison figures
- `artifacts/figures/robustness/` â€” robustness and adversarial behavior figures
- `artifacts/figures/concepts/` â€” pipeline and DERG concept visuals
- `artifacts/tables/benchmark/` â€” benchmark, dataset, calibration, and efficiency tables
- `artifacts/tables/analysis/` â€” robustness and error-analysis tables
- `artifacts/tables/statistics/` â€” confidence intervals and pairwise significance tables
- `artifacts/reports/summary/` â€” consolidated paper-facing summaries
- `artifacts/practical_experiments/` â€” operational triage, workload, source-shift, and case-study artifacts

## New Modules

| Module | Purpose |
|---|---|
| `src/supporting_analysis/statistical_testing.py` | Bootstrap CIs, McNemar's test, Friedman test |
| `src/supporting_analysis/explainability.py` | SHAP, evidence attribution, counterfactual analysis |
| `src/supporting_analysis/xai_analysis.py` | SHAP, attention rollout, integrated gradients, LIME, and representation probing |
| `src/supporting_analysis/deep_analysis.py` | Subgroup calibration, MC-dropout uncertainty, counterfactual search, error decomposition, and behavioral tests |
| `src/supporting_analysis/practical_experiments.py` | Real-world triage, source-shift audit, workload simulation, and case-study generation |
| `src/supporting_analysis/temporal_analysis.py` | CTI freshness scoring, temporal trust adjustment |
| `src/supporting_analysis/latex_export.py` | Publication-ready LaTeX tables |
| `src/supporting_analysis/case_study.py` | Representative case selection and analysis |

## Limitations

- TrustSOC-DERG requires PyTorch and may need GPU for efficient training.
- Expected action labels are heuristic targets derived from evidence rules, not human annotations.
- Temporal analysis uses heuristic half-life values; real CTI timestamps may vary.

## Future Work

- Enable a fully evaluated GNN encoder for true graph-based DERG reasoning
- Add human annotations for trust actions
- Rerun OpenSOC-AI and external baselines under the same evaluation harness
- Integrate real-time CTI feeds for temporal analysis validation

## Citation

Citation placeholder for the future paper.
