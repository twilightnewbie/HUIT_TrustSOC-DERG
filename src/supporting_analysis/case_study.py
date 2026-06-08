"""Case study selection and analysis for TrustSOC paper.

Selects representative cases covering all trust actions and adversarial types,
generates detailed per-case analysis with DERG visualization data,
and exports case study content suitable for academic papers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evidence_extractor import safe_text
from ..utils import save_json


# ---------------------------------------------------------------------------
# Case Selection Strategy
# ---------------------------------------------------------------------------

def select_representative_cases(
    test_df: pd.DataFrame,
    trust_scores: np.ndarray,
    expected_actions: np.ndarray,
    n_cases: int = 8,
) -> list[int]:
    """Select representative cases for paper case studies.
    
    Strategy:
    1. One case per action type (conclude, investigate, escalate, refuse)
    2. One borderline case (trust near threshold)
    3. One adversarial case that was correctly refused
    4. One adversarial case that was incorrectly trusted
    5. Fill remaining with diverse adversarial types
    """
    selected = []
    used_indices = set()

    # 1. One case per action type
    for action in ["conclude", "investigate", "escalate", "refuse"]:
        mask = expected_actions == action
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        # Pick case with most extreme trust score for that action
        if action in ("conclude",):
            idx = indices[np.argmax(trust_scores[indices])]
        elif action in ("refuse",):
            idx = indices[np.argmin(trust_scores[indices])]
        else:
            # Pick the most "typical" case (median trust)
            median_idx = np.argsort(trust_scores[indices])[len(indices) // 2]
            idx = indices[median_idx]
        if idx not in used_indices:
            selected.append(int(idx))
            used_indices.add(idx)

    # 2. Borderline case (trust ≈ 0.5)
    distances = np.abs(trust_scores - 0.5)
    for idx in np.argsort(distances):
        if idx not in used_indices:
            selected.append(int(idx))
            used_indices.add(idx)
            break

    # 3. Correctly refused adversarial
    adv_mask = test_df["adversarial_type"].ne("normal_case").to_numpy()
    refused_mask = expected_actions == "refuse"
    correct_refuse = adv_mask & refused_mask
    if correct_refuse.any():
        indices = np.where(correct_refuse)[0]
        for idx in indices:
            if idx not in used_indices:
                selected.append(int(idx))
                used_indices.add(idx)
                break

    # 4. Incorrectly trusted adversarial (if any)
    trusted_adv = adv_mask & (expected_actions == "conclude")
    if trusted_adv.any():
        indices = np.where(trusted_adv)[0]
        for idx in indices:
            if idx not in used_indices:
                selected.append(int(idx))
                used_indices.add(idx)
                break

    # 5. Fill remaining with diverse adversarial types
    if len(selected) < n_cases:
        adv_types = test_df["adversarial_type"].unique()
        for adv_type in adv_types:
            if adv_type == "normal_case":
                continue
            mask = test_df["adversarial_type"].eq(adv_type).to_numpy()
            indices = np.where(mask)[0]
            for idx in indices:
                if idx not in used_indices and len(selected) < n_cases:
                    selected.append(int(idx))
                    used_indices.add(idx)
                    break

    return selected[:n_cases]


# ---------------------------------------------------------------------------
# Detailed Case Analysis
# ---------------------------------------------------------------------------

def analyze_single_case(
    row: pd.Series,
    trust_score: float,
    expected_action: str,
    case_index: int,
) -> dict[str, Any]:
    """Generate detailed analysis for a single case."""
    analysis = {
        "index": case_index,
        "case_id": safe_text(row.get("case_id", f"case_{case_index}")),
        "split": safe_text(row.get("split", "test")),

        # Classification
        "threat_type": safe_text(row.get("threat_type", "")),
        "severity": safe_text(row.get("severity", "")),
        "label": safe_text(row.get("label", "")),
        "case_type": safe_text(row.get("case_type", "")),
        "adversarial_type": safe_text(row.get("adversarial_type", "normal_case")),

        # Trust layer
        "trust_score": trust_score,
        "expected_action": expected_action,
        "expected_action_target": safe_text(row.get("expected_action_target", "")),
        "action_correct": expected_action == safe_text(row.get("expected_action_target", "")),

        # Evidence metrics
        "risk_score": float(row.get("risk_score", 0.0)),
        "contradiction_score": float(row.get("contradiction_score", 0.0)),
        "reliability_score": float(row.get("reliability_score", 0.7)),
        "evidence_consistency": float(row.get("evidence_consistency", 1.0)),
        "adversarial_noise_score": float(row.get("adversarial_noise_score", 0.0)),
        "cti_match_count": int(row.get("cti_match_count", 0)),
        "mitre_count": int(row.get("mitre_count", 0)),
        "evidence_diversity": float(row.get("evidence_diversity", 0.0)),

        # DERG graph features
        "num_derg_nodes": float(row.get("num_derg_nodes", 0)),
        "num_derg_edges": float(row.get("num_derg_edges", 0)),
        "graph_density": float(row.get("graph_density", 0.0)),
        "graph_centrality_score": float(row.get("graph_centrality_score", 0.0)),

        # Text snippet (truncated for readability)
        "event_text_snippet": safe_text(row.get("event_text", ""))[:300],
    }

    # Generate narrative
    analysis["narrative"] = _generate_narrative(analysis)
    return analysis


def _generate_narrative(analysis: dict[str, Any]) -> str:
    """Generate academic-style narrative for a case study."""
    lines = []

    case_id = analysis["case_id"]
    action = analysis["expected_action"]
    trust = analysis["trust_score"]
    adv_type = analysis["adversarial_type"]

    lines.append(f"**Case {case_id}** ({adv_type})")
    lines.append(f"Trust Score: {trust:.4f} → Action: {action}")

    if action == "refuse":
        lines.append(
            f"The system correctly identified high contradiction "
            f"(score={analysis['contradiction_score']:.2f}) and adversarial noise "
            f"(score={analysis['adversarial_noise_score']:.2f}), leading to a refusal "
            f"to provide a definitive conclusion."
        )
    elif action == "escalate":
        lines.append(
            f"Despite moderate trust, the risk score ({analysis['risk_score']:.1f}) "
            f"combined with insufficient evidence consistency "
            f"({analysis['evidence_consistency']:.2f}) triggered escalation to a human analyst."
        )
    elif action == "investigate":
        lines.append(
            f"The model recommended further investigation based on "
            f"{analysis['cti_match_count']} CTI matches and {analysis['mitre_count']} "
            f"MITRE techniques, with evidence diversity of {analysis['evidence_diversity']:.1f}."
        )
    else:
        lines.append(
            f"High trust ({trust:.4f}) with low contradiction "
            f"({analysis['contradiction_score']:.2f}) and reliable evidence "
            f"({analysis['reliability_score']:.2f}) supported autonomous conclusion."
        )

    if analysis["action_correct"]:
        lines.append("✓ Action matched the expected ground truth.")
    else:
        lines.append(
            f"✗ Expected action was '{analysis['expected_action_target']}' "
            f"but model chose '{action}'."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Export Case Studies
# ---------------------------------------------------------------------------

def export_case_studies(
    test_df: pd.DataFrame,
    trust_scores: np.ndarray,
    expected_actions: np.ndarray,
    output_dir: Path,
    n_cases: int = 8,
) -> dict[str, Any]:
    """Select and export case studies for the paper.
    
    Outputs:
    - JSON file with structured case data
    - Markdown file with formatted narratives
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = select_representative_cases(test_df, trust_scores, expected_actions, n_cases)

    cases = []
    for idx in indices:
        case = analyze_single_case(
            test_df.iloc[idx],
            float(trust_scores[idx]),
            str(expected_actions[idx]),
            idx,
        )
        cases.append(case)

    # Save JSON
    save_json(output_dir / "case_studies.json", cases)

    # Generate markdown
    md_lines = ["# TrustSOC Case Studies\n"]
    for i, case in enumerate(cases, 1):
        md_lines.append(f"## Case Study {i}\n")
        md_lines.append(case["narrative"])
        md_lines.append("")
        md_lines.append(f"| Metric | Value |")
        md_lines.append(f"|--------|-------|")
        for key in ["trust_score", "contradiction_score", "reliability_score",
                     "evidence_consistency", "adversarial_noise_score", "cti_match_count",
                     "mitre_count", "num_derg_nodes", "graph_density"]:
            md_lines.append(f"| {key} | {case[key]} |")
        md_lines.append("")

    md_content = "\n".join(md_lines)
    (output_dir / "case_studies.md").write_text(md_content, encoding="utf-8")

    return {
        "n_cases": len(cases),
        "json_path": str(output_dir / "case_studies.json"),
        "markdown_path": str(output_dir / "case_studies.md"),
        "action_distribution": {
            action: sum(1 for c in cases if c["expected_action"] == action)
            for action in ["conclude", "investigate", "escalate", "refuse"]
        },
    }
