from __future__ import annotations

import math
import pickle
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    F = None
    # Mock classes/objects so the module can be imported without torch
    class nn:
        class Module:
            def __init__(self, *args, **kwargs):
                pass
            def register_buffer(self, *args, **kwargs):
                pass
            def to(self, *args, **kwargs):
                return self
            def eval(self, *args, **kwargs):
                return self
            def train(self, *args, **kwargs):
                return self
            def __call__(self, *args, **kwargs):
                return self
    class torch:
        class Tensor:
            pass
        @staticmethod
        def tensor(*args, **kwargs):
            return torch.Tensor()
        @staticmethod
        def device(*args, **kwargs):
            return "cpu"
        @staticmethod
        def manual_seed(*args, **kwargs):
            pass
        @staticmethod
        def cuda(*args, **kwargs):
            pass
        @staticmethod
        def load(*args, **kwargs):
            return {}
        @staticmethod
        def save(*args, **kwargs):
            pass
    class Dataset:
        pass
    class DataLoader:
        pass
    class WeightedRandomSampler:
        pass

from sklearn.metrics import accuracy_score
from ..calibration_metrics import calibration_summary
from ..config import ProjectConfig
from ..trust_calibration import build_meta_features, decide_actions, fit_trust_calibrator
from ..utils import get_logger, save_json, set_seed
from .model_utils import classification_metrics, regression_metrics


ACTION_LABELS = ["conclude", "investigate", "escalate", "refuse"]

TRUST_META_FEATURE_NAMES = [
    "confidence",
    "uncertainty",
    "reliability",
    "contradiction",
    "adversarial_noise",
    "risk_score",
    "cti_match_score",
    "evidence_consistency",
    "threat_margin",
    "severity_margin",
    "label_margin",
    "min_margin",
    "max_margin",
]

NUMERIC_COLUMNS = [
    "num_derg_nodes",
    "num_derg_edges",
    "graph_density",
    "avg_reliability",
    "max_reliability",
    "min_reliability",
    "reliability_std",
    "contradiction_score",
    "cti_match_score",
    "mitre_risk_score",
    "entity_risk_score",
    "evidence_diversity",
    "evidence_consistency",
    "adversarial_noise_score",
    "graph_centrality_score",
    "high_risk_node_ratio",
    "conflicting_evidence_ratio",
    "reliability_score",
    "cti_match_count",
    "mitre_count",
    "evidence_node_count",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_:\.\-/\$<>|=]+", str(text).lower())


def combine_text(row: pd.Series) -> str:
    event_text = str(row.get("event_text", ""))
    evidence_text = str(row.get("evidence_text", ""))
    return f"{event_text} [SEP] {evidence_text}"


def build_vocab(texts: list[str], vocab_size: int = 20000, min_freq: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common(vocab_size - 2):
        if freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_texts(texts: list[str], vocab: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.zeros((len(texts), max_len), dtype=np.int64)
    attention = np.zeros((len(texts), max_len), dtype=np.int64)
    unk_id = vocab["<unk>"]
    for idx, text in enumerate(texts):
        ids = [vocab.get(token, unk_id) for token in tokenize(text)[:max_len]]
        if not ids:
            ids = [unk_id]
        token_ids[idx, : len(ids)] = ids
        attention[idx, : len(ids)] = 1
    return token_ids, attention


class LabelEncoderMap:
    def __init__(self, values: list[str] | list[float]) -> None:
        unique = list(dict.fromkeys(values))
        self.classes_ = unique
        self.to_id = {item: idx for idx, item in enumerate(unique)}

    def transform(self, values: list[str] | list[float]) -> np.ndarray:
        return np.asarray([self.to_id[item] for item in values], dtype=np.int64)

    def inverse_transform(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray([self.classes_[int(idx)] for idx in ids])


class TrustSOCDataset(Dataset):
    def __init__(
        self,
        token_ids: np.ndarray,
        attention_mask: np.ndarray,
        numeric_features: np.ndarray,
        threat_labels: np.ndarray,
        severity_labels: np.ndarray,
        label_labels: np.ndarray,
        action_labels: np.ndarray,
        risk_targets: np.ndarray,
        risk_class_labels: np.ndarray,
    ) -> None:
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.numeric_features = torch.tensor(numeric_features, dtype=torch.float32)
        self.threat_labels = torch.tensor(threat_labels, dtype=torch.long)
        self.severity_labels = torch.tensor(severity_labels, dtype=torch.long)
        self.label_labels = torch.tensor(label_labels, dtype=torch.long)
        self.action_labels = torch.tensor(action_labels, dtype=torch.long)
        self.risk_targets = torch.tensor(risk_targets, dtype=torch.float32)
        self.risk_class_labels = torch.tensor(risk_class_labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "token_ids": self.token_ids[index],
            "attention_mask": self.attention_mask[index],
            "numeric_features": self.numeric_features[index],
            "threat_labels": self.threat_labels[index],
            "severity_labels": self.severity_labels[index],
            "label_labels": self.label_labels[index],
            "action_labels": self.action_labels[index],
            "risk_targets": self.risk_targets[index],
            "risk_class_labels": self.risk_class_labels[index],
        }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


if HAS_TORCH:
    class InspectableTransformerEncoderLayer(nn.Module):
        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
            batch_first: bool = False,
            layer_norm_eps: float = 1e-5,
        ) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
                batch_first=batch_first,
            )
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.activation_name = activation
            self.last_attn_weights: torch.Tensor | None = None

        def _activation(self, x: torch.Tensor) -> torch.Tensor:
            if self.activation_name == "gelu":
                return F.gelu(x)
            return F.relu(x)

        def forward(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            is_causal: bool = False,
        ) -> torch.Tensor:
            attn_kwargs = {
                "attn_mask": src_mask,
                "key_padding_mask": src_key_padding_mask,
                "need_weights": True,
                "average_attn_weights": False,
            }
            try:
                src2, attn_weights = self.self_attn(src, src, src, is_causal=is_causal, **attn_kwargs)
            except TypeError:
                src2, attn_weights = self.self_attn(src, src, src, **attn_kwargs)
            self.last_attn_weights = attn_weights.detach()

            src = self.norm1(src + self.dropout1(src2))
            feedforward = self.linear2(self.dropout(self._activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(feedforward))
            return src
else:
    class InspectableTransformerEncoderLayer(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, src, *args, **kwargs):
            return src


class TrustSOCTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        numeric_dim: int,
        threat_classes: int,
        severity_classes: int,
        label_classes: int,
        action_classes: int,
        risk_classes: int,
        risk_values: np.ndarray,
        embed_dim: int = 128,
        nhead: int = 4,
        ff_dim: int = 256,
        num_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position = PositionalEncoding(embed_dim, max_len=max_len)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = InspectableTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 128),
            nn.GELU(),
        )
        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.Sigmoid(),
        )
        self.shared = nn.Sequential(
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 160),
            nn.GELU(),
        )
        self.threat_head = nn.Linear(160, threat_classes)
        self.severity_head = nn.Linear(160, severity_classes)
        self.label_head = nn.Linear(160, label_classes)
        self.action_head = nn.Linear(160, action_classes)
        self.risk_class_head = nn.Sequential(
            nn.Linear(160 + severity_classes + label_classes, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, risk_classes),
        )
        self.risk_residual_head = nn.Sequential(
            nn.Linear(160 + severity_classes, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.risk_gate_head = nn.Sequential(
            nn.Linear(160 + severity_classes + label_classes, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.register_buffer("risk_values", torch.tensor(risk_values / 100.0, dtype=torch.float32))
        self.nhead = nhead

    def _encode_text(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(token_ids)
        x = self.position(x)
        x = self.embedding_dropout(x)
        key_padding_mask = attention_mask == 0
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        masked = encoded * attention_mask.unsqueeze(-1)
        mean_pool = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        max_source = encoded.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
        max_pool = max_source.max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return encoded, self.text_proj(pooled)

    def _mix_representations(
        self,
        text_repr: torch.Tensor,
        numeric_repr: torch.Tensor,
        gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.fusion_gate(torch.cat([text_repr, numeric_repr], dim=1)) if gate_override is None else gate_override
        mixed = torch.cat([gate * text_repr, (1.0 - gate) * numeric_repr], dim=1)
        return gate, self.shared(mixed)

    def _heads_from_shared(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        threat_logits = self.threat_head(fused)
        severity_logits = self.severity_head(fused)
        label_logits = self.label_head(fused)
        action_logits = self.action_head(fused)

        risk_context = torch.cat([fused, severity_logits, label_logits], dim=1)
        risk_class_logits = self.risk_class_head(risk_context)
        risk_probs = torch.softmax(risk_class_logits, dim=1)
        risk_from_class = (risk_probs * self.risk_values.unsqueeze(0)).sum(dim=1)

        risk_residual = 0.03 * torch.tanh(
            self.risk_residual_head(torch.cat([fused, severity_logits], dim=1)).squeeze(-1)
        )
        risk_gate = torch.sigmoid(self.risk_gate_head(risk_context)).squeeze(-1)
        risk_pred = torch.clamp(risk_from_class + risk_gate * risk_residual, 0.0, 1.0)
        return {
            "threat_logits": threat_logits,
            "severity_logits": severity_logits,
            "label_logits": label_logits,
            "action_logits": action_logits,
            "risk_class_logits": risk_class_logits,
            "risk_pred": risk_pred,
        }

    def forward_from_representations(
        self,
        text_repr: torch.Tensor,
        numeric_repr: torch.Tensor,
        gate_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        gate, fused = self._mix_representations(text_repr, numeric_repr, gate_override=gate_override)
        outputs = self._heads_from_shared(fused)
        outputs.update(
            {
                "gate": gate,
                "text_repr": text_repr,
                "numeric_repr": numeric_repr,
                "shared_repr": fused,
            }
        )
        return outputs

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor, numeric_features: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded, text_repr = self._encode_text(token_ids, attention_mask)
        numeric_repr = self.numeric_mlp(numeric_features)
        outputs = self.forward_from_representations(text_repr, numeric_repr)
        attention_weights = [
            layer.last_attn_weights
            for layer in self.encoder.layers
            if hasattr(layer, "last_attn_weights") and layer.last_attn_weights is not None
        ]
        outputs["encoded_tokens"] = encoded
        outputs["attention_weights"] = attention_weights
        return outputs


@dataclass
class EncodedBundle:
    train_dataset: TrustSOCDataset
    val_dataset: TrustSOCDataset
    test_dataset: TrustSOCDataset
    vocab: dict[str, int]
    threat_encoder: LabelEncoderMap
    severity_encoder: LabelEncoderMap
    label_encoder: LabelEncoderMap
    action_encoder: LabelEncoderMap
    risk_encoder: LabelEncoderMap
    numeric_mean: np.ndarray
    numeric_std: np.ndarray
    risk_values: np.ndarray


def build_encoded_bundle(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_vocab: int,
    max_len: int,
) -> EncodedBundle:
    train_texts = [combine_text(row) for _, row in train_df.iterrows()]
    val_texts = [combine_text(row) for _, row in val_df.iterrows()]
    test_texts = [combine_text(row) for _, row in test_df.iterrows()]
    vocab = build_vocab(train_texts, vocab_size=max_vocab, min_freq=2)
    train_tokens, train_mask = encode_texts(train_texts, vocab, max_len)
    val_tokens, val_mask = encode_texts(val_texts, vocab, max_len)
    test_tokens, test_mask = encode_texts(test_texts, vocab, max_len)

    threat_encoder = LabelEncoderMap(
        pd.concat([train_df["threat_type"], val_df["threat_type"], test_df["threat_type"]]).astype(str).tolist()
    )
    severity_encoder = LabelEncoderMap(
        pd.concat([train_df["severity"], val_df["severity"], test_df["severity"]]).astype(str).tolist()
    )
    label_encoder = LabelEncoderMap(
        pd.concat([train_df["label"], val_df["label"], test_df["label"]]).astype(str).tolist()
    )
    action_encoder = LabelEncoderMap(ACTION_LABELS)
    risk_values = np.sort(pd.concat([train_df["risk_score"], val_df["risk_score"], test_df["risk_score"]]).astype(float).unique()).astype(np.float32)
    risk_encoder = LabelEncoderMap(risk_values.tolist())

    train_numeric = train_df[NUMERIC_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    val_numeric = val_df[NUMERIC_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    test_numeric = test_df[NUMERIC_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    numeric_mean = train_numeric.mean(axis=0)
    numeric_std = train_numeric.std(axis=0)
    numeric_std[numeric_std < 1e-6] = 1.0
    train_numeric = (train_numeric - numeric_mean) / numeric_std
    val_numeric = (val_numeric - numeric_mean) / numeric_std
    test_numeric = (test_numeric - numeric_mean) / numeric_std

    train_risk_values = train_df["risk_score"].astype(float).to_numpy(dtype=np.float32)
    val_risk_values = val_df["risk_score"].astype(float).to_numpy(dtype=np.float32)
    test_risk_values = test_df["risk_score"].astype(float).to_numpy(dtype=np.float32)

    train_dataset = TrustSOCDataset(
        train_tokens,
        train_mask,
        train_numeric,
        threat_encoder.transform(train_df["threat_type"].astype(str).tolist()),
        severity_encoder.transform(train_df["severity"].astype(str).tolist()),
        label_encoder.transform(train_df["label"].astype(str).tolist()),
        action_encoder.transform(train_df["expected_action_target"].astype(str).tolist()),
        train_risk_values / 100.0,
        risk_encoder.transform(train_risk_values.tolist()),
    )
    val_dataset = TrustSOCDataset(
        val_tokens,
        val_mask,
        val_numeric,
        threat_encoder.transform(val_df["threat_type"].astype(str).tolist()),
        severity_encoder.transform(val_df["severity"].astype(str).tolist()),
        label_encoder.transform(val_df["label"].astype(str).tolist()),
        action_encoder.transform(val_df["expected_action_target"].astype(str).tolist()),
        val_risk_values / 100.0,
        risk_encoder.transform(val_risk_values.tolist()),
    )
    test_dataset = TrustSOCDataset(
        test_tokens,
        test_mask,
        test_numeric,
        threat_encoder.transform(test_df["threat_type"].astype(str).tolist()),
        severity_encoder.transform(test_df["severity"].astype(str).tolist()),
        label_encoder.transform(test_df["label"].astype(str).tolist()),
        action_encoder.transform(test_df["expected_action_target"].astype(str).tolist()),
        test_risk_values / 100.0,
        risk_encoder.transform(test_risk_values.tolist()),
    )

    return EncodedBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        vocab=vocab,
        threat_encoder=threat_encoder,
        severity_encoder=severity_encoder,
        label_encoder=label_encoder,
        action_encoder=action_encoder,
        risk_encoder=risk_encoder,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        risk_values=risk_values,
    )


def class_weight_tensor(encoded: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(encoded, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler(dataset: TrustSOCDataset, severity_classes: int) -> WeightedRandomSampler:
    threat_ids = dataset.threat_labels.numpy()
    severity_ids = dataset.severity_labels.numpy()
    combo = threat_ids * severity_classes + severity_ids
    counts = np.bincount(combo).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts[combo])
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def evaluate_model(
    model: TrustSOCTransformerModel,
    loader: DataLoader,
    device: torch.device,
    threat_encoder: LabelEncoderMap,
    severity_encoder: LabelEncoderMap,
    label_encoder: LabelEncoderMap,
    action_encoder: LabelEncoderMap,
    risk_encoder: LabelEncoderMap,
    collect_debug: bool = False,
) -> dict[str, Any]:
    model.eval()
    threat_logits_all = []
    severity_logits_all = []
    label_logits_all = []
    action_logits_all = []
    risk_class_logits_all = []
    risk_pred_all = []
    threat_true_all = []
    severity_true_all = []
    label_true_all = []
    action_true_all = []
    risk_true_all = []
    risk_class_true_all = []
    shared_repr_all = []
    gate_all = []
    text_repr_all = []
    numeric_repr_all = []
    attention_weights_all = []

    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            outputs = model(token_ids, attention_mask, numeric_features)
            threat_logits_all.append(outputs["threat_logits"].cpu())
            severity_logits_all.append(outputs["severity_logits"].cpu())
            label_logits_all.append(outputs["label_logits"].cpu())
            action_logits_all.append(outputs["action_logits"].cpu())
            risk_class_logits_all.append(outputs["risk_class_logits"].cpu())
            risk_pred_all.append(outputs["risk_pred"].cpu())
            threat_true_all.append(batch["threat_labels"])
            severity_true_all.append(batch["severity_labels"])
            label_true_all.append(batch["label_labels"])
            action_true_all.append(batch["action_labels"])
            risk_true_all.append(batch["risk_targets"])
            risk_class_true_all.append(batch["risk_class_labels"])
            if collect_debug:
                shared_repr_all.append(outputs["shared_repr"].cpu())
                gate_all.append(outputs["gate"].cpu())
                text_repr_all.append(outputs["text_repr"].cpu())
                numeric_repr_all.append(outputs["numeric_repr"].cpu())
                if outputs["attention_weights"]:
                    attention_weights_all.append(
                        [weights.cpu() for weights in outputs["attention_weights"]]
                    )

    threat_logits = torch.cat(threat_logits_all).numpy()
    severity_logits = torch.cat(severity_logits_all).numpy()
    label_logits = torch.cat(label_logits_all).numpy()
    action_logits = torch.cat(action_logits_all).numpy()
    risk_class_logits = torch.cat(risk_class_logits_all).numpy()
    risk_pred = np.clip(torch.cat(risk_pred_all).numpy() * 100.0, 0.0, 100.0)
    threat_true_ids = torch.cat(threat_true_all).numpy()
    severity_true_ids = torch.cat(severity_true_all).numpy()
    label_true_ids = torch.cat(label_true_all).numpy()
    action_true_ids = torch.cat(action_true_all).numpy()
    risk_class_true_ids = torch.cat(risk_class_true_all).numpy()
    risk_true = torch.cat(risk_true_all).numpy() * 100.0

    threat_pred_ids = threat_logits.argmax(axis=1)
    severity_pred_ids = severity_logits.argmax(axis=1)
    label_pred_ids = label_logits.argmax(axis=1)
    action_pred_ids = action_logits.argmax(axis=1)
    risk_class_pred_ids = risk_class_logits.argmax(axis=1)

    results = {
        "threat_true": threat_encoder.inverse_transform(threat_true_ids),
        "threat_pred": threat_encoder.inverse_transform(threat_pred_ids),
        "severity_true": severity_encoder.inverse_transform(severity_true_ids),
        "severity_pred": severity_encoder.inverse_transform(severity_pred_ids),
        "label_true": label_encoder.inverse_transform(label_true_ids),
        "label_pred": label_encoder.inverse_transform(label_pred_ids),
        "action_true": action_encoder.inverse_transform(action_true_ids),
        "action_pred": action_encoder.inverse_transform(action_pred_ids),
        "risk_true": risk_true,
        "risk_pred": risk_pred,
        "risk_class_true": risk_encoder.inverse_transform(risk_class_true_ids).astype(float),
        "risk_class_pred": risk_encoder.inverse_transform(risk_class_pred_ids).astype(float),
        "threat_logits": threat_logits,
        "severity_logits": severity_logits,
        "label_logits": label_logits,
        "action_logits": action_logits,
        "risk_class_logits": risk_class_logits,
    }
    if collect_debug:
        results["shared_repr"] = torch.cat(shared_repr_all).numpy() if shared_repr_all else np.empty((0, 160), dtype=np.float32)
        results["gate"] = torch.cat(gate_all).numpy() if gate_all else np.empty((0, 128), dtype=np.float32)
        results["text_repr"] = torch.cat(text_repr_all).numpy() if text_repr_all else np.empty((0, 128), dtype=np.float32)
        results["numeric_repr"] = torch.cat(numeric_repr_all).numpy() if numeric_repr_all else np.empty((0, 128), dtype=np.float32)
        results["attention_weights"] = attention_weights_all
    return results


def softmax_scores(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(np.clip(logits, -40.0, 40.0))
    probs = exps / exps.sum(axis=1, keepdims=True)
    ordered = np.sort(probs, axis=1)
    confidence = ordered[:, -1]
    uncertainty = 1.0 - confidence
    margin = ordered[:, -1] - ordered[:, -2] if ordered.shape[1] > 1 else confidence
    return probs, confidence, uncertainty if probs.shape[1] > 1 else np.zeros_like(confidence), margin


def build_transformer_meta(
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    reliability: np.ndarray,
    contradiction: np.ndarray,
    adversarial_noise: np.ndarray,
    risk_score: np.ndarray,
    cti_match_score: np.ndarray,
    evidence_consistency: np.ndarray,
    threat_margin: np.ndarray,
    severity_margin: np.ndarray,
    label_margin: np.ndarray,
) -> np.ndarray:
    base = build_meta_features(
        confidence,
        uncertainty,
        reliability,
        contradiction,
        adversarial_noise,
        risk_score,
        cti_match_score,
        evidence_consistency,
    )
    extra = np.column_stack(
        [
            threat_margin,
            severity_margin,
            label_margin,
            np.minimum.reduce([threat_margin, severity_margin, label_margin]),
            np.maximum.reduce([threat_margin, severity_margin, label_margin]),
        ]
    )
    return np.hstack([base, extra])


def compute_validation_score(outputs: dict[str, Any]) -> tuple[float, float, float, float]:
    joint = (
        (outputs["threat_true"] == outputs["threat_pred"])
        & (outputs["severity_true"] == outputs["severity_pred"])
        & (outputs["label_true"] == outputs["label_pred"])
    )
    joint_score = float(joint.mean())
    threat_f1 = classification_metrics(outputs["threat_true"], outputs["threat_pred"])["f1_weighted"]
    risk_mae = regression_metrics(outputs["risk_true"], outputs["risk_pred"])["mae"]
    risk_quality = max(0.0, 1.0 - min(risk_mae / 25.0, 1.0))
    val_score = 0.45 * joint_score + 0.30 * threat_f1 + 0.25 * risk_quality
    return val_score, joint_score, threat_f1, risk_mae


def train_derg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    model_name: str = "trustsoc_derg",
) -> dict[str, Any]:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed. Please install torch to train or evaluate the Transformer model.")
    logger = get_logger(config, f"train_{model_name}")
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_small_dataset = len(train_df) <= config.small_dataset_threshold
    use_warm_start = "warmstart" in model_name
    enable_sampler = "sampler" in model_name and "nosampler" not in model_name

    max_len = 144 if len(train_df) > 1000 or use_warm_start else 112
    max_vocab = 22000 if len(train_df) > 1000 or use_warm_start else 12000
    batch_size = 96 if device.type == "cuda" and not is_small_dataset else 32
    epochs = 8 if len(train_df) > 1000 else (12 if use_warm_start else 20)
    embed_dim = 128 if len(train_df) > 1000 or use_warm_start else 96
    ff_dim = 256 if len(train_df) > 1000 or use_warm_start else 192
    num_layers = 2 if len(train_df) > 1000 or use_warm_start else 1
    dropout = 0.10 if len(train_df) > 1000 or use_warm_start else 0.18

    bundle = build_encoded_bundle(train_df, val_df, test_df, max_vocab=max_vocab, max_len=max_len)
    sampler = build_weighted_sampler(bundle.train_dataset, len(bundle.severity_encoder.classes_)) if is_small_dataset and not use_warm_start and enable_sampler else None
    train_loader = DataLoader(bundle.train_dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None)
    val_loader = DataLoader(bundle.val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(bundle.test_dataset, batch_size=batch_size, shuffle=False)

    model = TrustSOCTransformerModel(
        vocab_size=len(bundle.vocab),
        numeric_dim=len(NUMERIC_COLUMNS),
        threat_classes=len(bundle.threat_encoder.classes_),
        severity_classes=len(bundle.severity_encoder.classes_),
        label_classes=len(bundle.label_encoder.classes_),
        action_classes=len(bundle.action_encoder.classes_),
        risk_classes=len(bundle.risk_encoder.classes_),
        risk_values=bundle.risk_values,
        embed_dim=embed_dim,
        nhead=4,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout,
    ).to(device)

    if use_warm_start:
        warm_start_path = config.models_dir / "trustsoc_derg" / "trustsoc_derg.pt"
        if warm_start_path.exists():
            checkpoint = torch.load(warm_start_path, map_location="cpu")
            current_state = model.state_dict()
            warm_state = checkpoint["state_dict"]
            compatible = {
                key: value
                for key, value in warm_state.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            current_state.update(compatible)
            model.load_state_dict(current_state)
            logger.info("Loaded %d warm-start tensors from %s", len(compatible), warm_start_path)

    threat_weights = class_weight_tensor(bundle.train_dataset.threat_labels.numpy(), len(bundle.threat_encoder.classes_), device)
    severity_weights = class_weight_tensor(bundle.train_dataset.severity_labels.numpy(), len(bundle.severity_encoder.classes_), device)
    label_weights = class_weight_tensor(bundle.train_dataset.label_labels.numpy(), len(bundle.label_encoder.classes_), device)
    action_weights = class_weight_tensor(bundle.train_dataset.action_labels.numpy(), len(bundle.action_encoder.classes_), device)
    risk_class_weights = class_weight_tensor(bundle.train_dataset.risk_class_labels.numpy(), len(bundle.risk_encoder.classes_), device)

    threat_loss_fn = nn.CrossEntropyLoss(weight=threat_weights)
    severity_loss_fn = nn.CrossEntropyLoss(weight=severity_weights)
    label_loss_fn = nn.CrossEntropyLoss(weight=label_weights)
    action_loss_fn = nn.CrossEntropyLoss(weight=action_weights)
    risk_class_loss_fn = nn.CrossEntropyLoss(weight=risk_class_weights)
    risk_loss_fn = nn.SmoothL1Loss()

    learning_rate = 2e-4 if not is_small_dataset else (1e-4 if use_warm_start else 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_rows = []
    best_state = None
    best_score = -1.0
    patience = 4 if not is_small_dataset else 6
    bad_epochs = 0
    train_start = time.perf_counter()

    if is_small_dataset and not use_warm_start:
        loss_weights = {
            "threat": 1.45,
            "severity": 0.95,
            "label": 0.35,
            "action": 0.20,
            "risk_class": 0.85,
            "risk_reg": 0.55,
        }
    elif is_small_dataset and use_warm_start:
        loss_weights = {
            "threat": 1.20,
            "severity": 0.90,
            "label": 0.30,
            "action": 0.20,
            "risk_class": 1.00,
            "risk_reg": 0.70,
        }
    else:
        loss_weights = {
            "threat": 1.25,
            "severity": 0.85,
            "label": 0.35,
            "action": 0.25,
            "risk_class": 1.30,
            "risk_reg": 0.90,
        }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            numeric_features = batch["numeric_features"].to(device)
            threat_labels = batch["threat_labels"].to(device)
            severity_labels = batch["severity_labels"].to(device)
            label_labels = batch["label_labels"].to(device)
            action_labels = batch["action_labels"].to(device)
            risk_targets = batch["risk_targets"].to(device)
            risk_class_labels = batch["risk_class_labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(token_ids, attention_mask, numeric_features)
                loss = (
                    loss_weights["threat"] * threat_loss_fn(outputs["threat_logits"], threat_labels)
                    + loss_weights["severity"] * severity_loss_fn(outputs["severity_logits"], severity_labels)
                    + loss_weights["label"] * label_loss_fn(outputs["label_logits"], label_labels)
                    + loss_weights["action"] * action_loss_fn(outputs["action_logits"], action_labels)
                    + loss_weights["risk_class"] * risk_class_loss_fn(outputs["risk_class_logits"], risk_class_labels)
                    + loss_weights["risk_reg"] * risk_loss_fn(outputs["risk_pred"], risk_targets)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())

        scheduler.step()
        val_outputs = evaluate_model(
            model,
            val_loader,
            device,
            bundle.threat_encoder,
            bundle.severity_encoder,
            bundle.label_encoder,
            bundle.action_encoder,
            bundle.risk_encoder,
        )
        val_score, val_joint, val_f1, val_risk_mae = compute_validation_score(val_outputs)
        history_rows.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(len(train_loader), 1),
                "accuracy": accuracy_score(val_outputs["threat_true"], val_outputs["threat_pred"]),
                "f1": val_f1,
                "joint_exact_match": val_joint,
                "risk_mae": val_risk_mae,
                "validation_score": val_score,
            }
        )
        logger.info(
            "Epoch %d/%d | loss=%.4f | val_score=%.4f | val_joint=%.4f | val_risk_mae=%.4f",
            epoch,
            epochs,
            history_rows[-1]["loss"],
            val_score,
            val_joint,
            val_risk_mae,
        )

        if val_score > best_score:
            best_score = val_score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    train_seconds = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    infer_start = time.perf_counter()
    train_outputs = evaluate_model(
        model,
        train_loader,
        device,
        bundle.threat_encoder,
        bundle.severity_encoder,
        bundle.label_encoder,
        bundle.action_encoder,
        bundle.risk_encoder,
    )
    val_outputs = evaluate_model(
        model,
        val_loader,
        device,
        bundle.threat_encoder,
        bundle.severity_encoder,
        bundle.label_encoder,
        bundle.action_encoder,
        bundle.risk_encoder,
    )
    test_outputs = evaluate_model(
        model,
        test_loader,
        device,
        bundle.threat_encoder,
        bundle.severity_encoder,
        bundle.label_encoder,
        bundle.action_encoder,
        bundle.risk_encoder,
    )
    inference_seconds = time.perf_counter() - infer_start

    train_correct = (
        (train_outputs["threat_true"] == train_outputs["threat_pred"])
        & (train_outputs["severity_true"] == train_outputs["severity_pred"])
        & (train_outputs["label_true"] == train_outputs["label_pred"])
    )
    val_correct = (
        (val_outputs["threat_true"] == val_outputs["threat_pred"])
        & (val_outputs["severity_true"] == val_outputs["severity_pred"])
        & (val_outputs["label_true"] == val_outputs["label_pred"])
    )
    test_correct = (
        (test_outputs["threat_true"] == test_outputs["threat_pred"])
        & (test_outputs["severity_true"] == test_outputs["severity_pred"])
        & (test_outputs["label_true"] == test_outputs["label_pred"])
    )

    _, train_conf_t, train_unc_t, train_margin_t = softmax_scores(train_outputs["threat_logits"])
    _, train_conf_s, train_unc_s, train_margin_s = softmax_scores(train_outputs["severity_logits"])
    _, train_conf_l, train_unc_l, train_margin_l = softmax_scores(train_outputs["label_logits"])
    _, val_conf_t, val_unc_t, val_margin_t = softmax_scores(val_outputs["threat_logits"])
    _, val_conf_s, val_unc_s, val_margin_s = softmax_scores(val_outputs["severity_logits"])
    _, val_conf_l, val_unc_l, val_margin_l = softmax_scores(val_outputs["label_logits"])
    _, test_conf_t, test_unc_t, test_margin_t = softmax_scores(test_outputs["threat_logits"])
    _, test_conf_s, test_unc_s, test_margin_s = softmax_scores(test_outputs["severity_logits"])
    _, test_conf_l, test_unc_l, test_margin_l = softmax_scores(test_outputs["label_logits"])

    train_conf = np.mean([train_conf_t, train_conf_s, train_conf_l], axis=0)
    val_conf = np.mean([val_conf_t, val_conf_s, val_conf_l], axis=0)
    test_conf = np.mean([test_conf_t, test_conf_s, test_conf_l], axis=0)
    train_unc = np.mean([train_unc_t, train_unc_s, train_unc_l], axis=0)
    val_unc = np.mean([val_unc_t, val_unc_s, val_unc_l], axis=0)
    test_unc = np.mean([test_unc_t, test_unc_s, test_unc_l], axis=0)

    train_meta = build_transformer_meta(
        train_conf,
        train_unc,
        train_df["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        train_df["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        train_df["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(train_outputs["risk_pred"], 0.0, 100.0) / 100.0,
        train_df["cti_match_score"].fillna(0.0).to_numpy(dtype=float),
        train_df["evidence_consistency"].fillna(1.0).to_numpy(dtype=float),
        train_margin_t,
        train_margin_s,
        train_margin_l,
    )
    val_meta = build_transformer_meta(
        val_conf,
        val_unc,
        val_df["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        val_df["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        val_df["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(val_outputs["risk_pred"], 0.0, 100.0) / 100.0,
        val_df["cti_match_score"].fillna(0.0).to_numpy(dtype=float),
        val_df["evidence_consistency"].fillna(1.0).to_numpy(dtype=float),
        val_margin_t,
        val_margin_s,
        val_margin_l,
    )
    test_meta = build_transformer_meta(
        test_conf,
        test_unc,
        test_df["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        test_df["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        test_df["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(test_outputs["risk_pred"], 0.0, 100.0) / 100.0,
        test_df["cti_match_score"].fillna(0.0).to_numpy(dtype=float),
        test_df["evidence_consistency"].fillna(1.0).to_numpy(dtype=float),
        test_margin_t,
        test_margin_s,
        test_margin_l,
    )

    split_point = max(16, len(val_df) // 2)
    split_point = min(split_point, len(val_df) - 1)
    calibrator, threshold, _ = fit_trust_calibrator(
        val_meta[:split_point],
        val_correct[:split_point],
        val_meta[split_point:],
        val_correct[split_point:],
        val_df["adversarial_type"].ne("normal_case").to_numpy()[split_point:],
        val_df["evidence_consistency"].fillna(1.0).to_numpy(dtype=float)[split_point:],
    )
    trust_score = calibrator.predict_proba(test_meta)[:, 1]
    expected_action_rule = decide_actions(
        trust_score,
        test_unc,
        test_df["avg_reliability"].fillna(0.7).to_numpy(dtype=float),
        test_df["contradiction_score"].fillna(0.0).to_numpy(dtype=float),
        test_df["adversarial_noise_score"].fillna(0.0).to_numpy(dtype=float),
        np.clip(test_outputs["risk_pred"], 0.0, 100.0) / 100.0,
    )
    calibration = calibration_summary(
        trust_score,
        test_correct.astype(bool),
        threshold,
        test_df["adversarial_type"].ne("normal_case").to_numpy(),
        test_df["evidence_consistency"].fillna(1.0).to_numpy(dtype=float),
    )
    calibration["threshold"] = float(threshold)
    refusal_mask = expected_action_rule == "refuse"
    escalation_mask = expected_action_rule == "escalate"
    calibration["refusal_accuracy"] = float(
        (test_df.loc[refusal_mask, "expected_action_target"] == "refuse").mean() if refusal_mask.any() else 0.0
    )
    calibration["escalation_accuracy"] = float(
        (test_df.loc[escalation_mask, "expected_action_target"] == "escalate").mean() if escalation_mask.any() else 0.0
    )
    corr = np.corrcoef(trust_score, test_outputs["risk_pred"])[0, 1]
    calibration["trust_risk_alignment"] = float(0.0 if np.isnan(corr) else corr)

    threat_metrics = classification_metrics(test_outputs["threat_true"], test_outputs["threat_pred"])
    severity_metrics = classification_metrics(test_outputs["severity_true"], test_outputs["severity_pred"])
    label_metrics = classification_metrics(test_outputs["label_true"], test_outputs["label_pred"])
    risk_metrics = regression_metrics(test_outputs["risk_true"], test_outputs["risk_pred"])
    risk_metrics["exact_accuracy"] = float(accuracy_score(test_outputs["risk_class_true"], test_outputs["risk_class_pred"]))

    predictions = pd.DataFrame(
        {
            "case_id": test_df["case_id"],
            "global_id": test_df["global_id"],
            "threat_true": test_outputs["threat_true"],
            "threat_pred": test_outputs["threat_pred"],
            "severity_true": test_outputs["severity_true"],
            "severity_pred": test_outputs["severity_pred"],
            "label_true": test_outputs["label_true"],
            "label_pred": test_outputs["label_pred"],
            "risk_true": test_outputs["risk_true"],
            "risk_pred": np.round(test_outputs["risk_pred"], 4),
            "risk_class_true": test_outputs["risk_class_true"],
            "risk_class_pred": test_outputs["risk_class_pred"],
            "trust_score": np.round(trust_score, 6),
            "uncertainty_score": np.round(test_unc, 6),
            "reliability_score": np.round(test_df["avg_reliability"].fillna(0.7).to_numpy(dtype=float), 6),
            "expected_action_pred": expected_action_rule,
            "expected_action_head": test_outputs["action_pred"],
            "expected_action_true": test_df["expected_action_target"],
            "joint_correct": test_correct,
            "adversarial_type": test_df["adversarial_type"],
            "event_text": test_df["event_text"],
            "evidence_text": test_df.get("evidence_text", pd.Series([""] * len(test_df))),
        }
    )

    metrics = {
        "model_name": model_name,
        "status": "trained",
        "dataset_rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "threat_type": threat_metrics,
        "severity": severity_metrics,
        "label": label_metrics,
        "risk_score": risk_metrics,
        "calibration": calibration,
        "joint_exact_match": float(test_correct.mean()),
        "expected_action_accuracy": float((predictions["expected_action_pred"] == predictions["expected_action_true"]).mean()),
        "expected_action_head_accuracy": float((predictions["expected_action_head"] == predictions["expected_action_true"]).mean()),
        "efficiency": {
            "train_time_seconds": float(train_seconds),
            "total_inference_seconds": float(inference_seconds),
            "average_latency_seconds_per_sample": float(inference_seconds / max(len(test_df), 1)),
            "feature_count": int(len(bundle.vocab) + len(NUMERIC_COLUMNS)),
            "parameter_count": int(sum(param.numel() for param in model.parameters())),
            "device": str(device),
        },
        "architecture": {
            "embed_dim": int(embed_dim),
            "ff_dim": int(ff_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "nhead": 4,
            "max_len": int(max_len),
        },
        "explainability": {
            "trust_meta_feature_names": TRUST_META_FEATURE_NAMES,
        },
    }

    predictions_path = config.predictions_dir / f"predictions_{model_name}.csv"
    metrics_path = config.metrics_dir / f"metrics_{model_name}.json"
    model_dir = config.models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.pt"
    bundle_path = model_dir / f"{model_name}_bundle.pkl"
    history_path = config.metrics_dir / f"training_history_{model_name}.csv"

    predictions.to_csv(predictions_path, index=False, encoding="utf-8")
    pd.DataFrame(history_rows).to_csv(history_path, index=False, encoding="utf-8")
    save_json(metrics_path, metrics)
    torch.save({"state_dict": model.state_dict()}, model_path)
    with bundle_path.open("wb") as handle:
        pickle.dump(
            {
                "vocab": bundle.vocab,
                "numeric_columns": NUMERIC_COLUMNS,
                "threat_classes": bundle.threat_encoder.classes_,
                "severity_classes": bundle.severity_encoder.classes_,
                "label_classes": bundle.label_encoder.classes_,
                "action_classes": bundle.action_encoder.classes_,
                "risk_classes": bundle.risk_encoder.classes_,
                "risk_values": bundle.risk_values.tolist(),
                "numeric_mean": bundle.numeric_mean,
                "numeric_std": bundle.numeric_std,
                "max_len": max_len,
                "architecture": metrics["architecture"],
                "trust_calibrator": calibrator,
                "trust_threshold": threshold,
                "trust_meta_feature_names": TRUST_META_FEATURE_NAMES,
                "metrics": metrics,
            },
            handle,
        )

    logger.info("Saved %s metrics to %s", model_name, metrics_path)
    return metrics


def train_tct_backbone(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ProjectConfig,
    model_name: str = "trustsoc_derg",
) -> dict[str, Any]:
    """Backward-compatible alias for the TrustSOC-DERG deep training entrypoint."""
    return train_derg(train_df, val_df, test_df, config, model_name=model_name)
