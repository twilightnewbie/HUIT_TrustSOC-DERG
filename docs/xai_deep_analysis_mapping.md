# TrustSOC XAI and Deep Analysis Mapping

This note maps five XAI techniques and five deep-analysis techniques to the current TrustSOC codebase.
It focuses on two goals:

1. Explain feature contribution for trust and action decisions.
2. Extend the current evaluation stack with publication-ready deep analysis.

## Existing anchors in the repo

- Trust calibrator: `src/trust_calibration.py`
- Calibration metrics: `src/calibration_metrics.py`
- Transformer hybrid model: `src/models/trustsoc_transformer.py`
- Existing explainability helpers: `src/supporting_analysis/explainability.py`
- Robustness benchmark: `src/robustness.py`
- Error analysis table generation: `src/error_analysis.py`
- Figure generation: `src/visualization.py`

## Five XAI techniques for feature contribution

### 1. SHAP on the trust calibrator

Best use:
- global feature contribution for publication
- local explanation of why a case is trusted or distrusted

Direct mapping:
- `fit_trust_calibrator()` in `src/trust_calibration.py`
- `build_transformer_meta()` in `src/models/trustsoc_transformer.py`
- `shap_trust_explanations()` in `src/supporting_analysis/explainability.py`

Why it fits this repo:
- the trust layer is a `LogisticRegression`, so SHAP is stable and fast
- the transformer trust meta-input is already explicit: 13 dimensions
  - 8 base trust features
  - 5 margin-derived features

Important interpretation detail:
- SHAP explains `trust_score`
- final action (`conclude`, `investigate`, `escalate`, `refuse`) is then produced by `decide_actions()`
- for a paper, the honest explanation is:
  1. SHAP explains why trust goes up or down
  2. rule tracing explains why that trust level and context produce `refuse` versus `conclude`

Recommended output:
- SHAP beeswarm for the 13 trust meta-features
- SHAP waterfall plot for representative `refuse` and `conclude` cases
- per-group SHAP comparison: `normal_case` vs adversarial

### 2. Attention rollout on the text encoder

Best use:
- token-level attribution from `event_text [SEP] evidence_text`
- showing whether the encoder focuses on alert text or supporting evidence

Direct mapping:
- `TrustSOCTransformerModel` in `src/models/trustsoc_transformer.py`
- `combine_text()` creates `event_text [SEP] evidence_text`

Why it fits this repo:
- the model uses `TransformerEncoderLayer` with `nhead=4`
- the architecture is already separated into text and numeric branches

Current limitation:
- the stock `TransformerEncoderLayer` does not expose attention weights in the current forward path
- a plain forward hook on `self.encoder` is not enough for reliable rollout

Needed change:
- replace the encoder layer with a custom wrapper that calls self-attention with `need_weights=True`
- save per-layer attention matrices
- rollout attention across both encoder layers
- split tokens before and after `[SEP]`

Recommended output:
- attention heatmap over tokens
- event-side attention mass vs evidence-side attention mass
- top attended tokens for trusted and refused cases

### 3. Integrated Gradients for text-vs-numeric fusion

Best use:
- explain whether a decision came more from text evidence or DERG numeric evidence
- support the Trust-Calibrated Transformer contribution

Direct mapping:
- `fusion_gate` in `src/models/trustsoc_transformer.py`
- text branch: `text_repr`
- numeric branch: `numeric_mlp`
- fused representation: `shared_repr`

Why it fits this repo:
- the model explicitly computes `gate * text_repr + (1 - gate) * numeric_repr`
- this is the cleanest place to explain source-of-evidence contribution

Current limitation:
- `gate` is computed in `forward()` but not returned

Needed change:
- either return `gate`, `text_repr`, and `numeric_repr`
- or register hooks on `fusion_gate` and the branch outputs
- then use Captum Integrated Gradients on:
  - `risk_pred`
  - `action_logits`
  - or a binary target such as `refuse` vs `conclude`

Recommended output:
- gate attribution score per sample
- average gate attribution by adversarial type
- text-dominant vs numeric-dominant distribution plots

### 4. LIME for adversarial debugging

Best use:
- debugging specific failure cases
- especially useful for `adversarial_type != "normal_case"`

Direct mapping:
- raw text inputs from `event_text` and `evidence_text`
- numeric inputs from `NUMERIC_COLUMNS` in `src/models/trustsoc_transformer.py`
- adversarial split metadata from preprocessing and predictions

Why it fits this repo:
- LIME is easy to use for local black-box inspection
- it is especially useful when SHAP gives a stable global story but a few adversarial cases still behave strangely

Recommended setup:
- `lime_text` for the combined text
- `lime_tabular` for the 21 numeric columns
- run both on the same case and compare which modality drives the failure

Recommended output:
- side-by-side local explanation for text and numeric features
- a curated table of adversarial cases where LIME and SHAP disagree

### 5. Probing classifiers on `shared_repr`

Best use:
- check whether the learned representation encodes trust-relevant structure
- interpret geometry rather than only feature importance

Direct mapping:
- `shared_repr` returned by the transformer `forward()`
- `trust_risk_alignment` already logged in transformer metrics

Why it fits this repo:
- the representation is already exposed by the model
- probes can test what the shared space encodes:
  - threat type
  - severity
  - expected action
  - adversarial type
  - trust bands

Current limitation:
- `evaluate_model()` currently does not collect `shared_repr`

Needed change:
- append `shared_repr` during evaluation
- train linear probes on frozen embeddings
- visualize with t-SNE or UMAP

Recommended output:
- probe accuracy table
- UMAP/t-SNE plots colored by `expected_action_target`, `adversarial_type`, and trust bands
- correlation between geometric clustering and `trust_risk_alignment`

## Five deep-analysis techniques for the problem

### 1. ECE plus subgroup reliability diagrams

Best use:
- core Trust Calibration contribution
- measure whether trust means the same thing on normal and adversarial cases

Direct mapping:
- `calibration_summary()` in `src/calibration_metrics.py`
- `plot_calibration()` in `src/visualization.py`

What to add:
- subgroup ECE for:
  - `normal_case`
  - adversarial overall
  - each adversarial subtype
- per-bin ECE contribution
- reliability diagrams split by subgroup

Why it matters:
- a single global ECE can hide the exact failure mode that matters most: over-trusting adversarial samples

### 2. MC Dropout uncertainty decomposition

Best use:
- separate aleatoric from epistemic uncertainty
- justify why high-uncertainty cases refuse more often

Direct mapping:
- dropout is already used in `TrustSOCTransformerModel`
- dropout can be increased in smaller-data experiments to surface higher epistemic uncertainty

What to add:
- run `model.train()` at inference time with gradients off
- perform `T=30` stochastic forward passes
- compute:
  - predictive entropy
  - expected entropy
  - epistemic = predictive entropy - expected entropy

Why it matters:
- if refused cases show high epistemic uncertainty, the refusal decision becomes much easier to defend in a paper

### 3. Counterfactual analysis with DiCE or constrained search

Best use:
- minimal changes needed to flip `refuse -> conclude`
- directly useful for human-AI trust alignment claims

Direct mapping:
- 21 numeric features from `NUMERIC_COLUMNS`
- existing counterfactual helper in `src/supporting_analysis/explainability.py` can serve as a starting point

What to add:
- search over actionable numeric dimensions only
- report minimal perturbations such as:
  - how much `avg_reliability` must increase
  - how much `contradiction_score` must decrease
  - whether CTI support alone is enough to change the decision

Why it matters:
- this gives a causal-style answer, not just descriptive importance

### 4. Error decomposition across multi-task heads

Best use:
- understand how threat, severity, label, risk, and action errors interact

Direct mapping:
- `predictions_{model_name}.csv`
- `src/error_analysis.py`

What to add:
- conditional metrics:
  - `P(severity_correct | threat_correct)`
  - `P(label_correct | threat_correct, severity_correct)`
  - `P(action_correct | trust_above_threshold)`
- 3D confusion tensor over:
  - threat
  - severity
  - label

Why it matters:
- joint models often fail in structured ways
- this can show whether trust errors are upstream classification errors or a separate calibration problem

### 5. Behavioral testing beyond the current robustness subsets

Best use:
- specification-level testing of trust behavior
- stronger than only reporting aggregate robustness metrics

Direct mapping:
- `run_robustness()` in `src/robustness.py`

What to add:
- entity substitution invariance
- semantic equivalence testing
- monotonicity checks:
  - increasing reliability should not reduce trust
  - decreasing contradiction should not reduce trust
- consistency checks:
  - adding supporting CTI should not increase refusal probability

Why it matters:
- this turns robustness into a test suite of expected trust behavior, which is much stronger for analysis sections

## Recommended priority order

If the goal is the best publication value with the smallest implementation cost, the order should be:

1. SHAP on the trust calibrator
2. Subgroup ECE plus reliability diagrams
3. Counterfactual analysis on the 21 numeric features
4. MC Dropout uncertainty decomposition
5. Probing on `shared_repr`

If the goal is debugging difficult adversarial failures, the order should be:

1. LIME
2. Counterfactual analysis
3. Behavioral testing
4. MC Dropout
5. Attention rollout

## Bottom line

For feature contribution, the strongest five-technique stack in this repo is:

1. SHAP
2. Attention rollout
3. Integrated Gradients
4. LIME
5. Probing classifiers

For deep problem analysis, the strongest five-technique stack is:

1. ECE plus subgroup reliability diagrams
2. MC Dropout uncertainty decomposition
3. Counterfactual or DiCE analysis
4. Error decomposition
5. Behavioral testing

The cleanest publication story is:
- SHAP explains trust feature contribution
- rule tracing explains the final action
- ECE and reliability diagrams validate trust calibration
- counterfactuals show what would change a refusal
- uncertainty decomposition explains why refusal is appropriate in high-uncertainty or adversarial settings
