from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

from .evidence_extractor import safe_text


def add_node(
    graph: nx.DiGraph,
    node_id: str,
    node_type: str,
    reliability_weight: float,
    confidence_score: float,
    contradiction_weight: float,
    source_type: str,
    evidence_role: str,
    risk_score: float,
) -> None:
    graph.add_node(
        node_id,
        node_type=node_type,
        reliability_weight=float(reliability_weight),
        confidence_score=float(confidence_score),
        contradiction_weight=float(contradiction_weight),
        source_type=source_type,
        evidence_role=evidence_role,
        risk_score=float(risk_score),
    )


def add_edge(
    graph: nx.DiGraph,
    source: str,
    target: str,
    relation: str,
    reliability_weight: float,
    contradiction_weight: float,
    source_type: str,
    evidence_role: str,
    risk_score: float,
) -> None:
    graph.add_edge(
        source,
        target,
        relation=relation,
        reliability_weight=float(reliability_weight),
        confidence_score=float(reliability_weight),
        contradiction_weight=float(contradiction_weight),
        source_type=source_type,
        evidence_role=evidence_role,
        risk_score=float(risk_score),
    )


def build_derg(case_row: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    case_id = safe_text(case_row.get("case_id", "unknown_case"))
    raw = case_row.get("raw_fields") or {}
    contradiction_score = float(case_row.get("contradiction_score", 0.0))
    reliability_score = float(case_row.get("reliability_score", 0.7))
    risk_score = float(case_row.get("risk_score", 0.0)) / 100.0

    alert_node = f"alert::{case_id}"
    incident_node = f"incident::{case_id}"
    add_node(graph, alert_node, "alert", reliability_score, reliability_score, contradiction_score, "alert", "primary", risk_score)
    add_node(graph, incident_node, "incident", reliability_score, reliability_score, contradiction_score, "incident", "primary", risk_score)
    add_edge(graph, alert_node, incident_node, "alert-incident", reliability_score, contradiction_score, "alert", "primary", risk_score)

    for key, node_type in (
        ("EntityType", "entity"),
        ("AccountName", "account"),
        ("DeviceName", "device"),
        ("IpAddress", "ip"),
        ("Url", "domain"),
        ("FileName", "file"),
    ):
        value = safe_text(raw.get(key))
        if not value:
            continue
        node_id = f"{node_type}::{value}"
        add_node(graph, node_id, node_type, reliability_score, reliability_score, contradiction_score, "raw_field", safe_text(raw.get("EvidenceRole", "related")), risk_score)
        add_edge(graph, incident_node, node_id, f"incident-{node_type}", reliability_score, contradiction_score, "raw_field", safe_text(raw.get("EvidenceRole", "related")), risk_score)

    for idx, item in enumerate(case_row.get("evidence_items", [])):
        evidence_node = f"evidence::{case_id}::{idx}"
        add_node(
            graph,
            evidence_node,
            "evidence",
            float(item.get("reliability_weight", reliability_score)),
            float(item.get("reliability_weight", reliability_score)),
            contradiction_score,
            safe_text(item.get("source_type", "evidence")),
            safe_text(item.get("evidence_role", "support")),
            risk_score,
        )
        add_edge(
            graph,
            evidence_node,
            incident_node,
            "evidence-supports-label",
            float(item.get("reliability_weight", reliability_score)),
            contradiction_score,
            safe_text(item.get("source_type", "evidence")),
            safe_text(item.get("evidence_role", "support")),
            risk_score,
        )

    for mitre in case_row.get("mitre_list", []):
        node_id = f"mitre::{mitre}"
        add_node(graph, node_id, "mitre", 0.80, 0.80, contradiction_score, "mitre", "attack-technique", risk_score)
        add_edge(graph, incident_node, node_id, "entity-mitre", 0.80, contradiction_score, "mitre", "attack-technique", risk_score)

    for idx, match in enumerate(case_row.get("cti_matches", [])):
        node_id = f"cti::{idx}::{safe_text(match.get('value'))}"
        reliability = float(match.get("reliability", 0.70))
        add_node(graph, node_id, "cti", reliability, reliability, contradiction_score, safe_text(match.get("source", "cti")), "threat-intelligence", risk_score)
        add_edge(graph, incident_node, node_id, "entity-cti", reliability, contradiction_score, safe_text(match.get("source", "cti")), "threat-intelligence", risk_score)

    if safe_text(case_row.get("timestamp")):
        timestamp_node = f"timestamp::{safe_text(case_row.get('timestamp'))}"
        add_node(graph, timestamp_node, "timestamp", 0.70, 0.70, 0.0, "timestamp", "temporal", risk_score)
        add_edge(graph, incident_node, timestamp_node, "temporal", 0.70, 0.0, "timestamp", "temporal", risk_score)

    if contradiction_score > 0.35:
        contradiction_node = f"contradiction::{case_id}"
        add_node(graph, contradiction_node, "contradiction", 0.40, 0.40, contradiction_score, "derived", "contradiction", risk_score)
        add_edge(graph, contradiction_node, incident_node, "evidence-contradicts-label", 0.40, contradiction_score, "derived", "contradiction", risk_score)

    return graph


def derg_features(graph: nx.DiGraph) -> dict[str, float]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    density = float(nx.density(graph)) if node_count > 1 else 0.0

    node_reliability = np.array([graph.nodes[node].get("reliability_weight", 0.0) for node in graph.nodes], dtype=float)
    contradiction = np.array([graph.nodes[node].get("contradiction_weight", 0.0) for node in graph.nodes], dtype=float)
    risk_values = np.array([graph.nodes[node].get("risk_score", 0.0) for node in graph.nodes], dtype=float)

    evidence_types = [graph.nodes[node].get("node_type", "unknown") for node in graph.nodes]
    evidence_diversity = len(set(evidence_types))
    evidence_consistency = float(np.clip(1.0 - contradiction.mean() if len(contradiction) else 1.0, 0.0, 1.0))
    adversarial_noise_score = float(np.clip(contradiction.max() if len(contradiction) else 0.0, 0.0, 1.0))

    if node_count > 1:
        centrality_values = list(nx.degree_centrality(graph).values())
        graph_centrality_score = float(np.mean(centrality_values))
    else:
        graph_centrality_score = 0.0

    cti_nodes = sum(1 for node in graph.nodes if graph.nodes[node].get("node_type") == "cti")
    mitre_nodes = sum(1 for node in graph.nodes if graph.nodes[node].get("node_type") == "mitre")
    entity_nodes = sum(
        1 for node in graph.nodes if graph.nodes[node].get("node_type") in {"entity", "account", "device", "ip", "domain", "file"}
    )
    contradiction_nodes = sum(1 for node in graph.nodes if graph.nodes[node].get("node_type") == "contradiction")
    high_risk_node_ratio = float((risk_values >= 0.75).mean()) if len(risk_values) else 0.0
    conflicting_evidence_ratio = float(contradiction_nodes / max(node_count, 1))

    return {
        "num_derg_nodes": float(node_count),
        "num_derg_edges": float(edge_count),
        "graph_density": density,
        "avg_reliability": float(node_reliability.mean()) if len(node_reliability) else 0.0,
        "max_reliability": float(node_reliability.max()) if len(node_reliability) else 0.0,
        "min_reliability": float(node_reliability.min()) if len(node_reliability) else 0.0,
        "reliability_std": float(node_reliability.std()) if len(node_reliability) else 0.0,
        "contradiction_score": float(contradiction.mean()) if len(contradiction) else 0.0,
        "cti_match_score": float(min(1.0, cti_nodes / 5.0)),
        "mitre_risk_score": float(min(1.0, mitre_nodes / 3.0)),
        "entity_risk_score": float(min(1.0, entity_nodes / 6.0)),
        "evidence_diversity": float(evidence_diversity),
        "evidence_consistency": evidence_consistency,
        "adversarial_noise_score": adversarial_noise_score,
        "graph_centrality_score": graph_centrality_score,
        "high_risk_node_ratio": high_risk_node_ratio,
        "conflicting_evidence_ratio": conflicting_evidence_ratio,
    }


def graph_for_case(case_row: dict[str, Any]) -> tuple[nx.DiGraph, dict[str, float]]:
    graph = build_derg(case_row)
    return graph, derg_features(graph)
