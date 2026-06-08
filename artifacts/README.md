# Artifacts Placeholder

This public GitHub upload keeps only lightweight, paper-facing summaries.

Excluded from upload:
- trained model weights
- prediction dumps
- generated figures
- XAI raw outputs
- deep-analysis raw outputs
- practical-experiment raw outputs
- logs

These artifacts can be regenerated locally with:

```powershell
python main.py --mode train_derg
python main.py --mode report
```

Optional support-only regeneration:

```powershell
python main.py --mode xai
python main.py --mode deep_analysis
python main.py --mode practical_experiments
```
