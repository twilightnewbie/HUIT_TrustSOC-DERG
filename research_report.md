# Bao cao Nghien cuu: TrustSOC-DERG

> Ban paper-facing cua repo nay chi giu mot mo hinh noi bo duy nhat: `TrustSOC-DERG`.
> `TrustSOC-DERG` la ten cong bo cua he thong ket hop DERG + Trust Calibration Transformer (TCT).
> XAI, deep analysis, va practical experiments chi dong vai tro phan tich ho tro.

---

## 1. Scope hien tai

Repo da duoc lam sach de chi con mot truc mo hinh:

- `TrustSOC-DERG`: mo hinh flagship duy nhat de train, evaluate, xai, deep analysis, va report.
- `OpenSOC-AI` va cac sklearn baselines: chi dung de so sanh benchmark ben ngoai.
- Khong con branch model noi bo lich su.
- Khong con mot nhanh mo hinh thu hai de ke chuyen paper-facing.

CLI chinh:

```powershell
python main.py --mode preprocess
python main.py --mode train_baselines
python main.py --mode train_derg
python main.py --mode evaluate
python main.py --mode robustness
python main.py --mode xai
python main.py --mode deep_analysis
python main.py --mode practical_experiments
python main.py --mode report
python main.py --mode full_analysis
```

---

## 2. Bon contributions chinh

### Contribution 1 - Dynamic Evidence Reliability Graph (DERG)

Muc tieu:
- mo hinh hoa bang chung SOC co cau truc thay vi xu ly van ban phang
- dua do tin cay, mau thuan, va mat do bang chung vao pipeline trust

Code lien quan:
- `src/derg_builder.py`
- `src/preprocessing.py`
- `src/models/model_utils.py`

Vai tro trong he thong:
- tao DERG-derived numeric evidence features
- bo sung tin hieu cau truc cho mo hinh flagship

### Contribution 2 - Trust Calibration Transformer (TCT)

Muc tieu:
- hop nhat text signal va DERG-derived evidence features
- du doan `threat_type`, `severity`, `label`, `expected_action`, `risk_score`
- hoc trust score de quyet dinh `conclude`, `investigate`, `escalate`, `refuse`

Code lien quan:
- `src/models/trustsoc_transformer.py`
- `src/trust_calibration.py`

Vai tro trong he thong:
- day la implementation cot loi cua `TrustSOC-DERG`
- TCT khong duoc trinh bay nhu mot model rieng trong repo/paper-facing pipeline nua

### Contribution 3 - Adversarial SOC Hallucination Dataset

Muc tieu:
- tao benchmark doi nghich cho bai toan trust-aware SOC reasoning
- danh gia mo hinh duoi cac tinh huong noise, poisoning, suppression, missing CTI/MITRE, va adversarial evidence

Code lien quan:
- `src/adversarial_generator.py`
- `src/robustness.py`
- `data/raw/`
- `data/processed/`

Vai tro trong he thong:
- cung cap cac split va robustness views phuc vu evaluation va practical experiments

### Contribution 4 - Human-AI Trust Alignment Metric

Muc tieu:
- do xem AI co biet khi nao nen tu ket luan va khi nao nen nhuong quyen cho analyst hay khong
- danh gia do thang hang giua trust score va correctness thuc te

Code lien quan:
- `src/calibration_metrics.py`
- `src/trust_calibration.py`
- `src/evaluation.py`

Vai tro trong he thong:
- dung de hoc threshold thich ung
- dung de bao cao quality cua quyet dinh trust-aware

---

## 3. Mo hinh flagship: TrustSOC-DERG

`TrustSOC-DERG` la pipeline duy nhat can nhac trong paper-facing codebase:

- Text input: `event_text [SEP] evidence_text`
- Numeric input: DERG-derived features, CTI/MITRE features, contradiction/noise/reliability signals
- Heads:
  - threat classification
  - severity classification
  - adversarial label classification
  - expected action classification
  - risk score regression
- Trust layer:
  - temperature scaling
  - adaptive threshold learning
  - action decision via trust-aware calibration

Noi cach khac:
- `DERG` la phan representation cua bang chung
- `TCT` la co che hoc sau va trust calibration
- ten he thong cong bo duy nhat van la `TrustSOC-DERG`

---

## 4. Snapshot ket qua hien tai

Nguon:
- `artifacts/metrics/metrics_trustsoc_derg.json`
- `artifacts/reports/summary/scopus_summary.md`

Tap du lieu:
- train: `23049`
- val: `2881`
- test: `2882`

Chi so chinh cua `TrustSOC-DERG`:

| Metric | Gia tri |
|---|---:|
| Threat accuracy | 0.9781 |
| Threat weighted F1 | 0.9804 |
| Severity accuracy | 0.9913 |
| Label accuracy | 0.9927 |
| Joint exact match | 0.9691 |
| Risk MAE | 0.3373 |
| Risk R2 | 0.9931 |
| Trust Alignment Score | 0.7361 |
| ECE | 0.1411 |
| Expected action accuracy | 0.7529 |
| Avg latency / sample | 0.00152 s |

Y nghia:
- mo hinh giu duoc performance phan loai cao
- regression risk rat sat gia tri that
- trust calibration du tot de dung cho bai toan trust-aware actioning

---

## 5. Supporting analysis

XAI va deep analysis chi dung de dien giai va kiem tra sau, khong phai contribution doc lap.

Artifact chinh:
- XAI summary: `artifacts/xai/trustsoc_derg/xai_summary.json`
- Deep analysis summary: `artifacts/deep_analysis/trustsoc_derg/deep_analysis_summary.json`
- Practical experiments: `artifacts/practical_experiments/trustsoc_derg/practical_experiments_summary.json`

Muc dich:
- giai thich trust feature contribution
- kiem tra calibration theo subgroup
- phan tich uncertainty, counterfactual, error decomposition, behavioral testing
- mo ta tac dong van hanh trong boi canh SOC

---

## 6. Clean repo policy

Tu thoi diem nay, repo duoc giu theo cac quy tac sau:

- chi mot model noi bo duy nhat: `TrustSOC-DERG`
- khong duy tri branch model noi bo cu
- khong duy tri report/notebook lich su nhac lai cac branch model cu
- XAI, deep analysis, practical experiments nam trong `src/supporting_analysis/` va duoc xem la phan tich ho tro

Neu can viet paper:
- phan methodology nen dung 4 contributions o tren
- phan experiments nen ke `TrustSOC-DERG` la flagship model
- phan explainability nen dat la supporting analysis cho Contribution 2 va Contribution 4
