from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ProjectConfig
from ..data_loader import load_processed_split
from ..models.trustsoc_transformer import (
    HAS_TORCH,
    TRUST_META_FEATURE_NAMES,
    TrustSOCTransformerModel,
    build_transformer_meta,
    combine_text,
    encode_texts,
    softmax_scores,
)

if HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader, Dataset
else:
    torch = None
    DataLoader = None
    Dataset = object


@dataclass
class TransformerAnalysisContext:
    config: ProjectConfig
    model_name: str
    device: Any
    bundle: dict[str, Any]
    model: Any
    threshold: float
    calibrator: Any
    split_name: str
    dataframe: pd.DataFrame


class AnalysisDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, attention_mask: np.ndarray, numeric_features: np.ndarray) -> None:
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "token_ids": self.token_ids[index],
            "attention_mask": self.attention_mask[index],
            "numeric_features": self.numeric_features[index],
        }


def default_transformer_model_name() -> str:
    return "trustsoc_derg"


def model_artifact_paths(config: ProjectConfig, model_name: str) -> dict[str, Path]:
    model_dir = config.models_dir / model_name
    return {
        "model": model_dir / f"{model_name}.pt",
        "bundle": model_dir / f"{model_name}_bundle.pkl",
    }


def load_transformer_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def infer_architecture(bundle: dict[str, Any], model_name: str) -> dict[str, Any]:
    architecture = dict(bundle.get("architecture", {}))
    if architecture:
        return architecture
    return {
        "embed_dim": 128,
        "ff_dim": 256,
        "num_layers": 2,
        "dropout": 0.10,
        "nhead": 4,
        "max_len": int(bundle.get("max_len", 144)),
    }


def instantiate_transformer(bundle: dict[str, Any], model_path: Path, model_name: str, device: Any) -> Any:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for transformer XAI and deep analysis.")
    architecture = infer_architecture(bundle, model_name)
    model = TrustSOCTransformerModel(
        vocab_size=len(bundle["vocab"]),
        numeric_dim=len(bundle["numeric_columns"]),
        threat_classes=len(bundle["threat_classes"]),
        severity_classes=len(bundle["severity_classes"]),
        label_classes=len(bundle["label_classes"]),
        action_classes=len(bundle["action_classes"]),
        risk_classes=len(bundle["risk_classes"]),
        risk_values=np.asarray(bundle["risk_values"], dtype=np.float32),
        embed_dim=int(architecture["embed_dim"]),
        nhead=int(architecture["nhead"]),
        ff_dim=int(architecture["ff_dim"]),
        num_layers=int(architecture["num_layers"]),
        max_len=int(architecture["max_len"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model


def load_analysis_split(config: ProjectConfig, split_name: str) -> pd.DataFrame:
    return load_processed_split(config, split_name)


def encode_dataframe_for_transformer(df: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    texts = [combine_text(row) for _, row in df.iterrows()]
    token_ids, attention_mask = encode_texts(texts, bundle["vocab"], int(bundle["max_len"]))
    numeric_columns = bundle["numeric_columns"]
    numeric_features = df[numeric_columns].fillna(0.0).to_numpy(dtype=np.float32)
    numeric_mean = np.asarray(bundle["numeric_mean"], dtype=np.float32)
    numeric_std = np.asarray(bundle["numeric_std"], dtype=np.float32)
    numeric_std[numeric_std < 1e-6] = 1.0
    numeric_features = (numeric_features - numeric_mean) / numeric_std
    return {
        "texts": texts,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "numeric_features": numeric_features,
    }


def build_analysis_loader(encoded: dict[str, Any], batch_size: int = 32) -> Any:
    dataset = AnalysisDataset(
        encoded["token_ids"],
        encoded["attention_mask"],
        encoded["numeric_features"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _append_tensor(store: list[Any], value: Any) -> None:
    if value is not None:
        store.append(value.detach().cpu())


def run_transformer_inference(
    model: Any,
    encoded: dict[str, Any],
    batch_size: int = 32,
    train_mode: bool = False,
    collect_debug: bool = False,
) -> dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for transformer XAI and deep analysis.")
    loader = build_analysis_loader(encoded, batch_size=batch_size)
    previous_mode = model.training
    model.train(train_mode)
    logits_store: dict[str, list[Any]] = {
        "threat_logits": [],
        "severity_logits": [],
        "label_logits": [],
        "action_logits": [],
        "risk_class_logits": [],
        "risk_pred": [],
    }
    debug_store: dict[str, list[Any]] = {
        "shared_repr": [],
        "gate": [],
        "text_repr": [],
        "numeric_repr": [],
    }

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["token_ids"].to(model.risk_values.device),
                batch["attention_mask"].to(model.risk_values.device),
                batch["numeric_features"].to(model.risk_values.device),
            )
            for key in logits_store:
                _append_tensor(logits_store[key], outputs[key])
            if collect_debug:
                for key in debug_store:
                    _append_tensor(debug_store[key], outputs[key])

    model.train(previous_mode)
    result = {
        key: torch.cat(values).numpy() if values else np.empty((0,), dtype=np.float32)
        for key, values in logits_store.items()
    }
    result["risk_pred"] = np.clip(result["risk_pred"] * 100.0, 0.0, 100.0)
    if collect_debug:
        result.update(
            {
                key: torch.cat(values).numpy() if values else np.empty((0,), dtype=np.float32)
                for key, values in debug_store.items()
            }
        )
    return result


def decode_predictions(logits: np.ndarray, classes: list[Any]) -> np.ndarray:
    if logits.ndim == 1:
        return np.asarray(classes)[(logits > 0).astype(int)]
    return np.asarray(classes)[logits.argmax(axis=1)]


def compute_transformer_meta(df: pd.DataFrame, outputs: dict[str, Any]) -> dict[str, Any]:
    _, conf_t, unc_t, margin_t = softmax_scores(outputs["threat_logits"])
    _, conf_s, unc_s, margin_s = softmax_scores(outputs["severity_logits"])
    _, conf_l, unc_l, margin_l = softmax_scores(outputs["label_logits"])
    confidence = np.mean([conf_t, conf_s, conf_l], axis=0)
    uncertainty = np.mean([unc_t, unc_s, unc_l], axis=0)
    meta = build_transformer_meta(
        confidence,
        uncertainty,
        df["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        df["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        df["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(outputs["risk_pred"], 0.0, 100.0) / 100.0,
        df["cti_match_score"].fillna(0.0).to_numpy(dtype=float),
        df["evidence_consistency"].fillna(1.0).to_numpy(dtype=float),
        margin_t,
        margin_s,
        margin_l,
    )
    return {
        "meta": meta,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "margins": {
            "threat_margin": margin_t,
            "severity_margin": margin_s,
            "label_margin": margin_l,
        },
        "feature_names": list(TRUST_META_FEATURE_NAMES),
    }


def load_transformer_analysis_context(
    config: ProjectConfig,
    model_name: str,
    split_name: str = "test",
) -> TransformerAnalysisContext:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for transformer XAI and deep analysis.")
    paths = model_artifact_paths(config, model_name)
    if not paths["bundle"].exists() or not paths["model"].exists():
        raise FileNotFoundError(
            f"Missing transformer artifacts for '{model_name}'. Expected {paths['bundle']} and {paths['model']}. "
            "Run `python main.py --mode train_derg` first."
        )
    bundle = load_transformer_bundle(paths["bundle"])
    calibrator = bundle.get("trust_calibrator")
    if calibrator is None:
        raise RuntimeError(
            "This transformer bundle does not include a saved trust calibrator. "
            "Retrain the transformer with the updated code before running XAI or deep analysis."
        )
    dataframe = load_analysis_split(config, split_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_transformer(bundle, paths["model"], model_name, device)
    return TransformerAnalysisContext(
        config=config,
        model_name=model_name,
        device=device,
        bundle=bundle,
        model=model,
        threshold=float(bundle.get("trust_threshold", bundle.get("metrics", {}).get("calibration", {}).get("threshold", 0.5))),
        calibrator=calibrator,
        split_name=split_name,
        dataframe=dataframe,
    )
