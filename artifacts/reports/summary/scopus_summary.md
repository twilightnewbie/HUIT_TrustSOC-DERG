# TrustSOC Research Summary

## Main Model
- TrustSOC-DERG: threat acc 0.9781, weighted F1 0.9804, risk MAE 0.3373, ECE 0.1411.

## XAI Status
- XAI suite status: completed
- XAI detail: artifacts written under artifacts/xai/

## Deep Analysis Status
- Deep-analysis suite status: completed
- Deep-analysis detail: artifacts written under artifacts/deep_analysis/

## Practical Experiment Status
- Practical experiment suite status: completed
- Clean-test manual reviews saved per 1000 alerts: 762.6648160999306
- Noisy-evidence refusal rate: 0.0006939625260235947

## Modeling Note
- TrustSOC-DERG is the sole flagship model in this repository, implementing the Trust Calibration Transformer over DERG-derived evidence features.

## Suggested Paper Positioning
- Present `TrustSOC-DERG` as the sole flagship model in the paper-facing pipeline.
- Use the XAI and deep-analysis artifacts as the main explanation and trust-analysis evidence for the paper.

## Main Figures
- Full TrustSOC-DERG training loss: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\figures\models\trustsoc_derg\training_loss_curve.png`
- Full TrustSOC-DERG validation overview: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\figures\models\trustsoc_derg\training_metrics_overview.png`
- Full TrustSOC-DERG calibration: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\figures\models\trustsoc_derg\calibration_curve.png`
- Full TrustSOC-DERG risk scatter: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\figures\models\trustsoc_derg\risk_true_vs_predicted.png`

## Key Artifacts
- Main training history: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\metrics\training_history_trustsoc_derg.csv`
- XAI summary: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\xai\trustsoc_derg\xai_summary.json`
- Deep-analysis summary: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\deep_analysis\trustsoc_derg\deep_analysis_summary.json`
- Practical experiments summary: `C:\Users\Tai\Downloads\trustsoc-research-main\TrustSOC-DERG\artifacts\practical_experiments\trustsoc_derg\practical_experiments_summary.json`