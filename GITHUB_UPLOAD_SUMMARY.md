# GitHub Upload Summary

This repository has been slimmed down for public upload.

## What is included

- source code under `src/`
- runnable entrypoints under `main.py` and `scripts/`
- paper-facing documentation in `README.md` and `research_report.md`
- compact result summaries in `artifacts/metrics/`, `artifacts/tables/`, and `artifacts/reports/summary/`

## What is intentionally excluded

- raw and processed datasets
- trained model weights
- prediction dumps
- generated figures
- XAI raw outputs
- deep-analysis raw outputs
- practical-experiment raw outputs
- logs and old zip bundles

## Flagship model

- Public model name: `TrustSOC-DERG`
- Main implementation: `src/models/trustsoc_transformer.py`
- Public training entrypoint: `train_derg(...)` in `src/models/trustsoc_transformer.py`

## Verified public snapshot

The cleaned repository was verified after refactoring and cleanup with:

```powershell
python -m compileall src main.py scripts/train_derg.py scripts/train_transformer.py
python main.py --mode report
```

## Main reported results

- Threat accuracy: `0.9781`
- Threat weighted F1: `0.9804`
- Severity accuracy: `0.9913`
- Joint exact match: `0.9691`
- Risk MAE: `0.3373`
- Risk R2: `0.9931`
- Trust Alignment Score: `0.7361`
- ECE: `0.1411`
- Manual reviews saved per 1000 clean alerts: `762.66`

## Reproduction note

To regenerate omitted heavy assets locally, restore the datasets and run:

```powershell
python main.py --mode preprocess
python main.py --mode train_derg
python main.py --mode report
python main.py --mode xai
python main.py --mode deep_analysis
python main.py --mode practical_experiments
```
