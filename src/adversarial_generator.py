"""Adversarial SOC Hallucination Benchmark for TrustSOC.

Generates robustness evaluation subsets with a comprehensive attack taxonomy:
1. Noise Injection — adds benign noise text
2. Evidence Poisoning — injects fake high-reliability CTI
3. Evidence Suppression — removes critical evidence
4. Label Manipulation — contradicts evidence labels
5. Missing CTI — zeroes CTI features
6. Missing MITRE — zeroes MITRE features
7. Noisy Evidence — adds distracting benign signals
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Adversarial Type Assignment
# ---------------------------------------------------------------------------

def assign_adversarial_type(row: pd.Series) -> str:
    text = str(row.get("event_text", "")).lower()
    if row.get("case_type") == "synthetic_adversarial_case":
        if "injected_adversarial_signal=" in text and "benignpositive" in text:
            return "contradictory_evidence"
        if "page=http" in text or "include=http" in text:
            return "hallucination_trap"
        if "<script" in text:
            return "noisy_evidence"
        return "synthetic_adversarial_case"
    if float(row.get("contradiction_score", 0.0)) >= 0.50:
        return "conflicting_verdict"
    if int(row.get("cti_match_count", 0)) == 0 and str(row.get("label")) == "adversarial":
        return "missing_cti_evidence"
    if str(row.get("mitre_techniques", "UNKNOWN")).upper() == "UNKNOWN" and str(row.get("label")) == "adversarial":
        return "mitre_mismatch"
    if len(str(row.get("evidence_text", ""))) < 80:
        return "incomplete_evidence"
    return "normal_case"


# ---------------------------------------------------------------------------
# Expected Action Assignment
# ---------------------------------------------------------------------------

def assign_expected_action(row: pd.Series) -> str:
    contradiction = float(row.get("contradiction_score", 0.0))
    risk = float(row.get("risk_score", 0.0))
    noise = float(row.get("adversarial_noise_score", 0.0))
    consistency = float(row.get("evidence_consistency", 1.0))

    if contradiction >= 0.60 or noise >= 0.70:
        return "refuse"
    if risk >= 75 and (contradiction >= 0.35 or consistency <= 0.55):
        return "escalate"
    if contradiction >= 0.30 or consistency <= 0.70:
        return "investigate"
    return "conclude"


# ---------------------------------------------------------------------------
# Original Robustness Views (preserved)
# ---------------------------------------------------------------------------

def create_robustness_views(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create original + new adversarial robustness evaluation subsets."""
    adversarial = test_df[test_df["adversarial_type"] != "normal_case"].copy()

    missing_cti = test_df.copy()
    missing_cti["cti_matches_json"] = "[]"
    missing_cti["cti_match_count"] = 0
    missing_cti["cti_match_score"] = 0.0
    missing_cti["expected_action_target"] = missing_cti.apply(assign_expected_action, axis=1)

    missing_mitre = test_df.copy()
    missing_mitre["mitre_techniques"] = "UNKNOWN"
    missing_mitre["mitre_list_json"] = "[]"
    missing_mitre["mitre_count"] = 0
    missing_mitre["mitre_risk_score"] = 0.0
    missing_mitre["expected_action_target"] = missing_mitre.apply(assign_expected_action, axis=1)

    noisy = test_df.copy()
    noisy["event_text"] = noisy["event_text"].astype(str) + " | benign_noise=health-check-ok | benign_noise=no-malware-found"
    noisy["adversarial_type"] = "noisy_evidence"
    noisy["adversarial_noise_score"] = np.clip(noisy["adversarial_noise_score"].astype(float) + 0.20, 0.0, 1.0)
    noisy["expected_action_target"] = noisy.apply(assign_expected_action, axis=1)

    # New attack types
    poisoned = create_evidence_poisoning(test_df)
    suppressed = create_evidence_suppression(test_df)
    label_flipped = create_label_manipulation(test_df)

    return {
        "adversarial": adversarial.reset_index(drop=True),
        "missing_cti": missing_cti.reset_index(drop=True),
        "missing_mitre": missing_mitre.reset_index(drop=True),
        "noisy_evidence": noisy.reset_index(drop=True),
        "evidence_poisoning": poisoned.reset_index(drop=True),
        "evidence_suppression": suppressed.reset_index(drop=True),
        "label_manipulation": label_flipped.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# NEW: Evidence Poisoning Attack
# ---------------------------------------------------------------------------

def create_evidence_poisoning(test_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject fake high-reliability CTI matches to mislead the model.
    
    Attack strategy: add convincing-looking CTI matches that shouldn't
    be there, testing if the model blindly trusts fake evidence.
    """
    rng = np.random.RandomState(seed)
    poisoned = test_df.copy()

    fake_cti_templates = [
        {"type": "ip", "value": "10.0.0.1", "reliability": 0.95, "source": "fake_high_reliability"},
        {"type": "domain", "value": "benign-update.example.com", "reliability": 0.92, "source": "fake_high_reliability"},
        {"type": "cve", "value": "CVE-2024-99999", "reliability": 0.88, "source": "fake_high_reliability"},
    ]

    for idx in range(len(poisoned)):
        # Parse existing CTI matches
        cti_str = str(poisoned.iloc[idx].get("cti_matches_json", "[]"))
        try:
            existing = json.loads(cti_str)
        except (json.JSONDecodeError, TypeError):
            existing = []

        # Inject 1-2 fake CTI matches
        n_fake = rng.randint(1, 3)
        for _ in range(n_fake):
            fake = fake_cti_templates[rng.randint(0, len(fake_cti_templates))].copy()
            fake["value"] = fake["value"] + f"_{rng.randint(1000, 9999)}"
            existing.append(fake)

        poisoned.at[poisoned.index[idx], "cti_matches_json"] = json.dumps(existing, ensure_ascii=False)
        poisoned.at[poisoned.index[idx], "cti_match_count"] = len(existing)
        poisoned.at[poisoned.index[idx], "cti_match_score"] = min(1.0, 0.20 * len(existing))

    poisoned["adversarial_type"] = "evidence_poisoning"
    poisoned["adversarial_noise_score"] = np.clip(
        poisoned["adversarial_noise_score"].astype(float) + 0.25, 0.0, 1.0
    )
    poisoned["expected_action_target"] = poisoned.apply(assign_expected_action, axis=1)

    return poisoned


# ---------------------------------------------------------------------------
# NEW: Evidence Suppression Attack
# ---------------------------------------------------------------------------

def create_evidence_suppression(test_df: pd.DataFrame) -> pd.DataFrame:
    """Remove critical evidence to test model behavior with incomplete information.
    
    Attack strategy: strip CTI, MITRE, and evidence items simultaneously,
    leaving only raw text. Tests if the model recognizes insufficient evidence.
    """
    suppressed = test_df.copy()

    # Remove all structured evidence
    suppressed["cti_matches_json"] = "[]"
    suppressed["cti_match_count"] = 0
    suppressed["cti_match_score"] = 0.0
    suppressed["mitre_techniques"] = "UNKNOWN"
    suppressed["mitre_list_json"] = "[]"
    suppressed["mitre_count"] = 0
    suppressed["mitre_risk_score"] = 0.0
    suppressed["evidence_items_json"] = "[]"

    # Reduce evidence-derived features
    suppressed["evidence_diversity"] = 1.0
    suppressed["evidence_consistency"] = 0.5
    suppressed["reliability_score"] = 0.5

    # Update graph features
    suppressed["num_derg_nodes"] = 2.0  # Only alert + incident nodes remain
    suppressed["num_derg_edges"] = 1.0
    suppressed["graph_density"] = 1.0

    suppressed["adversarial_type"] = "evidence_suppression"
    suppressed["adversarial_noise_score"] = np.clip(
        suppressed["adversarial_noise_score"].astype(float) + 0.15, 0.0, 1.0
    )
    suppressed["expected_action_target"] = suppressed.apply(assign_expected_action, axis=1)

    return suppressed


# ---------------------------------------------------------------------------
# NEW: Label Manipulation Attack
# ---------------------------------------------------------------------------

def create_label_manipulation(test_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject contradictory labels to test model's contradiction detection.
    
    Attack strategy: for adversarial cases, flip the incident grade to
    look benign; for benign cases, add adversarial markers. Tests if the
    trust layer detects label-evidence inconsistency.
    """
    rng = np.random.RandomState(seed)
    manipulated = test_df.copy()

    for idx in range(len(manipulated)):
        current_label = str(manipulated.iloc[idx].get("label", ""))
        event_text = str(manipulated.iloc[idx].get("event_text", ""))

        if current_label == "adversarial":
            # Make adversarial cases look benign
            manipulated.at[manipulated.index[idx], "event_text"] = (
                event_text + " | system_verdict=BenignPositive | risk_level=Low"
            )
        else:
            # Make benign cases look adversarial
            if rng.random() < 0.5:
                manipulated.at[manipulated.index[idx], "event_text"] = (
                    event_text + " | injected_adversarial_signal=critical_threat_detected"
                )

    # Increase contradiction scores
    manipulated["contradiction_score"] = np.clip(
        manipulated["contradiction_score"].astype(float) + 0.30, 0.0, 1.0
    )
    manipulated["evidence_consistency"] = np.clip(
        manipulated["evidence_consistency"].astype(float) - 0.25, 0.0, 1.0
    )
    manipulated["adversarial_type"] = "label_manipulation"
    manipulated["adversarial_noise_score"] = np.clip(
        manipulated["adversarial_noise_score"].astype(float) + 0.20, 0.0, 1.0
    )
    manipulated["expected_action_target"] = manipulated.apply(assign_expected_action, axis=1)

    return manipulated


# ---------------------------------------------------------------------------
# Attack Taxonomy Summary
# ---------------------------------------------------------------------------

ATTACK_TAXONOMY = {
    "noise_injection": {
        "description": "Adds benign noise text to event logs",
        "target": "Text features",
        "expected_impact": "Model should maintain accuracy despite noise",
        "function": "noisy_evidence in create_robustness_views",
    },
    "evidence_poisoning": {
        "description": "Injects fake high-reliability CTI matches",
        "target": "CTI matching and reliability scoring",
        "expected_impact": "Trust layer should detect inflated reliability",
        "function": "create_evidence_poisoning",
    },
    "evidence_suppression": {
        "description": "Removes all structured evidence, leaving only raw text",
        "target": "DERG construction, evidence features",
        "expected_impact": "Model should refuse/investigate due to insufficient evidence",
        "function": "create_evidence_suppression",
    },
    "label_manipulation": {
        "description": "Flips evidence labels to create contradiction",
        "target": "Contradiction detection, trust calibration",
        "expected_impact": "Trust layer should detect label-evidence inconsistency",
        "function": "create_label_manipulation",
    },
    "missing_cti": {
        "description": "Zeros all CTI features",
        "target": "CTI matching pipeline",
        "expected_impact": "Model should downgrade confidence without CTI",
        "function": "missing_cti in create_robustness_views",
    },
    "missing_mitre": {
        "description": "Zeros all MITRE ATT&CK features",
        "target": "MITRE technique mapping",
        "expected_impact": "Model should increase investigation recommendation",
        "function": "missing_mitre in create_robustness_views",
    },
    "adversarial_cases": {
        "description": "Naturally occurring adversarial samples from dataset",
        "target": "Full pipeline",
        "expected_impact": "Model should refuse or escalate adversarial cases",
        "function": "adversarial filter in create_robustness_views",
    },
}
