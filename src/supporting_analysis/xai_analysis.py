from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .analysis_support import (
    compute_transformer_meta,
    encode_dataframe_for_transformer,
    load_transformer_analysis_context,
    run_transformer_inference,
)
from .explainability import shap_trust_explanations
from ..models.trustsoc_transformer import tokenize
from ..trust_calibration import decide_actions
from ..utils import get_logger, save_json

sns.set_theme(style="whitegrid")


def _safe_case_indices(df: pd.DataFrame, actions: np.ndarray) -> list[int]:
    indices: list[int] = []
    for action in ("refuse", "conclude"):
        mask = np.where(actions == action)[0]
        if len(mask):
            indices.append(int(mask[0]))
    adv_mask = np.where(df["adversarial_type"].ne("normal_case").to_numpy())[0]
    if len(adv_mask):
        indices.append(int(adv_mask[0]))
    if not indices and len(df):
        indices.append(0)
    return list(dict.fromkeys(indices))


def _save_barplot(series: pd.Series, out_path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    series.sort_values().plot.barh(ax=ax, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_shap_feature_analysis(
    calibrator: Any,
    meta_features: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_payload = shap_trust_explanations(
        calibrator,
        meta_features,
        feature_names=feature_names,
        max_samples=min(300, len(meta_features)),
    )
    if shap_payload is None:
        coefs = pd.Series(calibrator.coef_.flatten(), index=feature_names)
        coefs.to_csv(output_dir / "shap_fallback_coefficients.csv", header=["coefficient"])
        _save_barplot(
            coefs.abs().sort_values(ascending=False).head(13),
            output_dir / "shap_fallback_importance.png",
            "Trust Calibrator Coefficient Importance",
            "Absolute coefficient",
        )
        return {
            "mode": "coefficient_fallback",
            "top_features": coefs.abs().sort_values(ascending=False).head(5).index.tolist(),
        }

    save_json(output_dir / "shap_summary.json", shap_payload)
    importance = pd.Series(shap_payload["feature_importance"])
    importance.to_csv(output_dir / "shap_feature_importance.csv", header=["mean_abs_shap"])
    _save_barplot(
        importance.head(13),
        output_dir / "shap_feature_importance.png",
        "Mean Absolute SHAP Importance",
        "Mean |SHAP|",
    )

    try:
        import shap

        sample = meta_features[: min(300, len(meta_features))]
        shap_values = np.asarray(shap_payload["shap_values"])
        plt.figure(figsize=(11, 6))
        shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    return {
        "mode": "shap",
        "top_features": list(importance.head(5).index),
    }


def attention_rollout(attention_weights: list[Any], attention_mask: np.ndarray) -> np.ndarray:
    if not attention_weights:
        return np.empty((0,), dtype=np.float32)
    rollout = None
    valid_length = int(attention_mask.sum())
    identity = None
    for layer_weights in attention_weights:
        weights = layer_weights.squeeze(0).mean(dim=0).cpu().numpy()
        weights = weights[:valid_length, :valid_length]
        if identity is None:
            identity = np.eye(weights.shape[0], dtype=np.float32)
        weights = weights + identity
        weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-6, None)
        rollout = weights if rollout is None else rollout @ weights
    return rollout[0]


def run_attention_analysis(
    context,
    encoded: dict[str, Any],
    case_indices: list[int],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for idx in case_indices:
        token_ids = encoded["token_ids"][idx : idx + 1]
        attention_mask = encoded["attention_mask"][idx : idx + 1]
        numeric_features = encoded["numeric_features"][idx : idx + 1]
        with __import__("torch").no_grad():
            outputs = context.model(
                __import__("torch").tensor(token_ids, dtype=__import__("torch").long, device=context.device),
                __import__("torch").tensor(attention_mask, dtype=__import__("torch").long, device=context.device),
                __import__("torch").tensor(numeric_features, dtype=__import__("torch").float32, device=context.device),
            )
        rollout = attention_rollout(outputs["attention_weights"], attention_mask[0])
        tokens = tokenize(encoded["texts"][idx])
        valid_length = min(len(tokens), len(rollout))
        token_scores = pd.Series(rollout[:valid_length], index=tokens[:valid_length]).sort_values(ascending=False)
        token_scores.head(20).to_csv(output_dir / f"attention_case_{idx}.csv", header=["rollout"])

        fig, ax = plt.subplots(figsize=(10, 4))
        token_scores.head(15).sort_values().plot.barh(ax=ax, color="#dc2626")
        ax.set_title(f"Attention Rollout Case {idx}")
        ax.set_xlabel("Attention contribution")
        fig.tight_layout()
        fig.savefig(output_dir / f"attention_case_{idx}.png", dpi=300)
        plt.close(fig)

        sep_pos = next((i for i, token in enumerate(tokens[:valid_length]) if token.lower() == "sep"), valid_length // 2)
        event_mass = float(np.sum(rollout[: min(sep_pos, valid_length)]))
        evidence_mass = float(np.sum(rollout[min(sep_pos + 1, valid_length) : valid_length]))
        summaries.append(
            {
                "case_index": idx,
                "event_attention_mass": event_mass,
                "evidence_attention_mass": evidence_mass,
                "top_tokens": token_scores.head(8).index.tolist(),
            }
        )
    save_json(output_dir / "attention_summary.json", summaries)
    return summaries


def integrated_gradients_branch_contributions(
    context,
    encoded: dict[str, Any],
    case_indices: list[int],
    output_dir: Path,
    steps: int = 24,
) -> pd.DataFrame:
    torch = __import__("torch")
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in case_indices:
        token_ids = torch.tensor(encoded["token_ids"][idx : idx + 1], dtype=torch.long, device=context.device)
        attention_mask = torch.tensor(encoded["attention_mask"][idx : idx + 1], dtype=torch.long, device=context.device)
        numeric_features = torch.tensor(encoded["numeric_features"][idx : idx + 1], dtype=torch.float32, device=context.device)

        with torch.no_grad():
            base_outputs = context.model(token_ids, attention_mask, numeric_features)
        text_repr = base_outputs["text_repr"].detach()
        numeric_repr = base_outputs["numeric_repr"].detach()
        predicted_action_idx = int(base_outputs["action_logits"].argmax(dim=1).item())
        predicted_action = context.bundle["action_classes"][predicted_action_idx]

        text_grad_sum = torch.zeros_like(text_repr)
        numeric_grad_sum = torch.zeros_like(numeric_repr)
        for alpha in torch.linspace(0.0, 1.0, steps, device=context.device):
            interp_text = (alpha * text_repr).detach().clone().requires_grad_(True)
            interp_numeric = (alpha * numeric_repr).detach().clone().requires_grad_(True)
            outputs = context.model.forward_from_representations(interp_text, interp_numeric)
            target = outputs["action_logits"][:, predicted_action_idx].sum()
            context.model.zero_grad(set_to_none=True)
            target.backward()
            text_grad_sum += interp_text.grad.detach()
            numeric_grad_sum += interp_numeric.grad.detach()

        text_attr = (text_repr * text_grad_sum / steps).abs().sum().item()
        numeric_attr = (numeric_repr * numeric_grad_sum / steps).abs().sum().item()
        total_attr = max(text_attr + numeric_attr, 1e-6)
        rows.append(
            {
                "case_index": idx,
                "target_action": predicted_action,
                "text_attribution": float(text_attr),
                "numeric_attribution": float(numeric_attr),
                "text_ratio": float(text_attr / total_attr),
                "numeric_ratio": float(numeric_attr / total_attr),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "integrated_gradients_branch_contributions.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    melted = frame.melt(id_vars=["case_index", "target_action"], value_vars=["text_ratio", "numeric_ratio"], var_name="branch", value_name="ratio")
    sns.barplot(data=melted, x="case_index", y="ratio", hue="branch", ax=ax, palette="deep")
    ax.set_title("Integrated Gradients Branch Contribution Ratios")
    ax.set_ylabel("Attribution ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "integrated_gradients_branch_contributions.png", dpi=300)
    plt.close(fig)
    return frame


def run_lime_analysis(
    context,
    encoded: dict[str, Any],
    case_indices: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from lime.lime_tabular import LimeTabularExplainer
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return {"status": "skipped", "reason": "lime_not_installed"}

    torch = __import__("torch")
    text_explainer = LimeTextExplainer(class_names=["not_refuse", "refuse"])
    tabular_explainer = LimeTabularExplainer(
        context.dataframe[context.bundle["numeric_columns"]].fillna(0.0).to_numpy(dtype=np.float32),
        feature_names=context.bundle["numeric_columns"],
        class_names=["not_refuse", "refuse"],
        discretize_continuous=True,
    )

    reports: list[dict[str, Any]] = []
    fixed_numeric = context.dataframe[context.bundle["numeric_columns"]].fillna(0.0).to_numpy(dtype=np.float32)

    for idx in case_indices:
        row = context.dataframe.iloc[idx]
        if row.get("adversarial_type", "normal_case") == "normal_case":
            continue
        text_value = encoded["texts"][idx]
        numeric_row = fixed_numeric[idx]
        refuse_idx = context.bundle["action_classes"].index("refuse")

        def predict_refuse_from_text(texts: list[str], numeric_row: np.ndarray = numeric_row, row: pd.Series = row) -> np.ndarray:
            sample_df = pd.DataFrame(
                {
                    "event_text": texts,
                    "evidence_text": [""] * len(texts),
                    **{col: [row.get(col, 0.0)] * len(texts) for col in context.bundle["numeric_columns"]},
                }
            )
            for col_idx, col in enumerate(context.bundle["numeric_columns"]):
                sample_df[col] = numeric_row[col_idx]
            encoded_local = encode_dataframe_for_transformer(sample_df, context.bundle)
            outputs = run_transformer_inference(context.model, encoded_local, batch_size=min(16, len(texts)))
            action_probs = np.exp(outputs["action_logits"] - outputs["action_logits"].max(axis=1, keepdims=True))
            action_probs = action_probs / np.clip(action_probs.sum(axis=1, keepdims=True), 1e-6, None)
            refuse_prob = action_probs[:, refuse_idx]
            return np.column_stack([1.0 - refuse_prob, refuse_prob])

        def predict_refuse_from_numeric(numeric_rows: np.ndarray, text_value: str = text_value, row: pd.Series = row) -> np.ndarray:
            sample_df = pd.DataFrame(
                {
                    "event_text": [row.get("event_text", text_value)] * len(numeric_rows),
                    "evidence_text": [row.get("evidence_text", "")] * len(numeric_rows),
                }
            )
            for col_idx, col in enumerate(context.bundle["numeric_columns"]):
                sample_df[col] = numeric_rows[:, col_idx]
            encoded_local = encode_dataframe_for_transformer(sample_df, context.bundle)
            outputs = run_transformer_inference(context.model, encoded_local, batch_size=min(16, len(numeric_rows)))
            action_probs = np.exp(outputs["action_logits"] - outputs["action_logits"].max(axis=1, keepdims=True))
            action_probs = action_probs / np.clip(action_probs.sum(axis=1, keepdims=True), 1e-6, None)
            refuse_prob = action_probs[:, refuse_idx]
            return np.column_stack([1.0 - refuse_prob, refuse_prob])

        text_exp = text_explainer.explain_instance(
            text_value,
            lambda texts, numeric_row=numeric_row: predict_refuse_from_text(texts, numeric_row),
            num_features=10,
        )
        tabular_exp = tabular_explainer.explain_instance(
            numeric_row,
            lambda arr, text_value=text_value: predict_refuse_from_numeric(arr, text_value),
            num_features=10,
        )
        report = {
            "case_index": idx,
            "adversarial_type": row.get("adversarial_type", "normal_case"),
            "text_explanation": text_exp.as_list(label=1),
            "tabular_explanation": tabular_exp.as_list(label=1),
        }
        reports.append(report)
        save_json(output_dir / f"lime_case_{idx}.json", report)
    save_json(output_dir / "lime_summary.json", reports)
    return {"status": "ok", "cases": len(reports)}


def run_probe_analysis(
    dataframe: pd.DataFrame,
    shared_repr: np.ndarray,
    trust_scores: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "expected_action": dataframe["expected_action_target"].astype(str).to_numpy(),
        "adversarial_type": dataframe["adversarial_type"].astype(str).to_numpy(),
        "threat_type": dataframe["threat_type"].astype(str).to_numpy(),
        "trust_band": np.asarray(pd.cut(trust_scores, bins=[-0.01, 0.4, 0.7, 1.0], labels=["low", "mid", "high"]).astype(str)),
    }

    rows: list[dict[str, Any]] = []
    for name, target in targets.items():
        if len(np.unique(target)) < 2:
            continue
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                shared_repr,
                target,
                test_size=0.3,
                random_state=42,
                stratify=target,
            )
        except ValueError:
            continue
        probe = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        probe.fit(X_train, y_train)
        pred = probe.predict(X_test)
        rows.append(
            {
                "probe_target": name,
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1_macro": float(f1_score(y_test, pred, average="macro")),
            }
        )

    probe_df = pd.DataFrame(rows)
    probe_df.to_csv(output_dir / "probe_metrics.csv", index=False, encoding="utf-8")

    if len(shared_repr) > 2:
        perplexity = max(1, min(30, max(1, len(shared_repr) // 3)))
        coords = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(shared_repr)
        projection_name = "tsne"

        plot_df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "expected_action": dataframe["expected_action_target"].astype(str).to_numpy(),
                "adversarial_type": dataframe["adversarial_type"].astype(str).to_numpy(),
            }
        )
        plot_df.to_csv(output_dir / f"{projection_name}_projection.csv", index=False, encoding="utf-8")
        for column in ("expected_action", "adversarial_type"):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x="x", y="y", hue=column, s=35, ax=ax)
            ax.set_title(f"Shared Representation {projection_name.upper()} colored by {column}")
            fig.tight_layout()
            fig.savefig(output_dir / f"{projection_name}_{column}.png", dpi=300)
            plt.close(fig)

    return {
        "probe_targets": probe_df["probe_target"].tolist() if not probe_df.empty else [],
    }


def run_xai_suite(config, model_name: str = "trustsoc_derg", fail_on_missing: bool = True) -> dict[str, Any]:
    logger = get_logger(config, "xai")
    output_dir = config.xai_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        context = load_transformer_analysis_context(config, model_name=model_name)
    except Exception as exc:
        summary = {
            "model_name": model_name,
            "status": "skipped",
            "reason": str(exc),
        }
        save_json(output_dir / "xai_summary.json", summary)
        logger.warning("Skipping XAI suite for %s: %s", model_name, exc)
        if fail_on_missing:
            raise
        return summary

    encoded = encode_dataframe_for_transformer(context.dataframe, context.bundle)
    outputs = run_transformer_inference(context.model, encoded, batch_size=32, collect_debug=True)
    meta_payload = compute_transformer_meta(context.dataframe, outputs)
    trust_scores = context.calibrator.predict_proba(meta_payload["meta"])[:, 1]
    actions = decide_actions(
        trust_scores,
        meta_payload["uncertainty"],
        context.dataframe["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        context.dataframe["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        context.dataframe["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(outputs["risk_pred"], 0.0, 100.0) / 100.0,
    )
    case_indices = _safe_case_indices(context.dataframe, actions)
    logger.info("Selected XAI case indices: %s", case_indices)

    shap_summary = run_shap_feature_analysis(
        context.calibrator,
        meta_payload["meta"],
        meta_payload["feature_names"],
        output_dir / "shap",
    )
    attention_summary = run_attention_analysis(context, encoded, case_indices, output_dir / "attention")
    ig_frame = integrated_gradients_branch_contributions(context, encoded, case_indices, output_dir / "integrated_gradients")
    lime_summary = run_lime_analysis(context, encoded, case_indices, output_dir / "lime")
    probe_summary = run_probe_analysis(context.dataframe, outputs["shared_repr"], trust_scores, output_dir / "probes")

    summary = {
        "model_name": model_name,
        "n_samples": int(len(context.dataframe)),
        "threshold": float(context.threshold),
        "selected_cases": case_indices,
        "shap": shap_summary,
        "attention": attention_summary,
        "integrated_gradients_cases": ig_frame.to_dict(orient="records"),
        "lime": lime_summary,
        "probes": probe_summary,
    }
    save_json(output_dir / "xai_summary.json", summary)
    logger.info("Saved XAI outputs to %s", output_dir)
    return summary
