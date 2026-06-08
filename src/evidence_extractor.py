from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_RELIABILITY = {
    "guide_train.csv": 0.78,
    "guide_test.csv": 0.78,
    "1_otx_threat_intel.csv": 0.73,
    "2_cve_vulnerabilities.csv": 0.86,
    "3_malicious_domains.csv": 0.84,
    "4_malicious_ips.csv": 0.84,
}

THREAT_PATTERNS = {
    "BRUTE FORCE": [r"failed password", r"password spray", r"credential stuffing", r"t1110"],
    "SQL INJECTION": [r"union select", r" or 1=1", r"information_schema", r"sleep\(", r"t1190"],
    "XSS": [r"<script", r"javascript:", r"onerror=", r"alert\(", r"t1189"],
    "PATH TRAVERSAL": [r"\.\./", r"\.\.\\", r"/etc/passwd", r"boot\.ini", r"path traversal"],
    "COMMAND INJECTION": [r"cmd=", r"\|sh", r"powershell", r"cmd\.exe", r"wget http", r"t1059"],
    "REMOTE FILE INCLUSION": [r"page=http", r"include=http"],
    "LOG4J JNDI": [r"\$\{jndi:", r"ldap://", r"rmi://"],
    "SCANNER": [r"nmap", r"masscan", r"nikto", r"reconnaissance", r"t1595"],
    "DATA EXFILTRATION": [r"exfiltration", r"data exfil", r"t1041", r"t1020"],
    "MALWARE": [r"malware", r"trojan", r"ransom", r"loader", r"payload"],
    "PRIVILEGE ESCALATION": [r"privilege escalation", r"sudo", r"setuid", r"t1068"],
    "PHISHING": [r"phish", r"attachment", r"macro", r"clickfix", r"t1566"],
    "VULNERABILITY": [r"cve-\d{4}-\d+", r"vulnerability", r"cwe-"],
}


def safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def find_all(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text, flags=re.IGNORECASE)


def build_cti_catalogs(raw_datasets: dict[str, pd.DataFrame]) -> dict[str, Any]:
    domain_df = raw_datasets.get("3_malicious_domains.csv", pd.DataFrame())
    ip_df = raw_datasets.get("4_malicious_ips.csv", pd.DataFrame())
    cve_df = raw_datasets.get("2_cve_vulnerabilities.csv", pd.DataFrame())
    otx_df = raw_datasets.get("1_otx_threat_intel.csv", pd.DataFrame())

    otx_index: list[dict[str, Any]] = []
    for _, row in otx_df.head(500).iterrows():
        attack_ids = safe_text(row.get("Attack_IDs"))
        description = safe_text(row.get("Description")).lower()
        title = safe_text(row.get("Title")).lower()
        tags = safe_text(row.get("Tags")).lower()
        otx_index.append(
            {
                "title": safe_text(row.get("Title")),
                "description": safe_text(row.get("Description")),
                "attack_ids": attack_ids,
                "match_text": " ".join([title, description, tags]),
            }
        )

    return {
        "malicious_domains": set(domain_df.get("Domain", pd.Series(dtype=str)).astype(str).str.lower()),
        "malicious_ips": set(ip_df.get("IP", pd.Series(dtype=str)).astype(str)),
        "cves": set(cve_df.get("cveID", pd.Series(dtype=str)).astype(str).str.upper()),
        "otx_records": otx_index,
    }


def extract_ips(text: str) -> list[str]:
    return sorted(set(find_all(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)))


def extract_domains(text: str) -> list[str]:
    candidates = find_all(r"\b[a-z0-9][a-z0-9\-\.]+\.[a-z]{2,}\b", text.lower())
    return sorted(set(candidates))


def extract_cves(text: str) -> list[str]:
    return sorted(set(item.upper() for item in find_all(r"cve-\d{4}-\d+", text)))


def extract_mitre_list(value: str) -> list[str]:
    tokens = re.findall(r"T\d{4}(?:\.\d{3})?", safe_text(value).upper())
    return sorted(set(tokens))


def rule_signal_summary(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {threat: sum(1 for pattern in patterns if re.search(pattern, lowered)) for threat, patterns in THREAT_PATTERNS.items()}


def derive_contradiction_score(row: dict[str, Any], rule_signals: dict[str, int], cti_match_count: int) -> float:
    raw = row.get("raw_fields") or {}
    incident_grade = safe_text(raw.get("IncidentGrade", "")).lower()
    total_rule_hits = sum(rule_signals.values())
    contradiction = 0.0

    if "benignpositive" in incident_grade and total_rule_hits > 0:
        contradiction += 0.35
    if "falsepositive" in incident_grade and safe_text(row.get("label")).lower() == "adversarial":
        contradiction += 0.20
    if safe_text(row.get("mitre_techniques")).upper() == "UNKNOWN" and total_rule_hits > 0:
        contradiction += 0.15
    if cti_match_count == 0 and safe_text(row.get("label")).lower() == "adversarial":
        contradiction += 0.10
    if "injected_adversarial_signal=" in safe_text(row.get("event_text")).lower():
        contradiction += 0.20

    return float(np.clip(contradiction, 0.0, 1.0))


def match_cti(row: dict[str, Any], catalogs: dict[str, Any]) -> list[dict[str, Any]]:
    text = safe_text(row.get("event_text"))
    lowered = text.lower()
    ips = extract_ips(text)
    domains = extract_domains(text)
    cves = extract_cves(text)
    mitre_list = extract_mitre_list(safe_text(row.get("mitre_techniques")))

    matches: list[dict[str, Any]] = []
    for ip in ips:
        if ip in catalogs["malicious_ips"]:
            matches.append({"type": "ip", "value": ip, "reliability": 0.84, "source": "4_malicious_ips.csv"})
    for domain in domains:
        if domain in catalogs["malicious_domains"]:
            matches.append({"type": "domain", "value": domain, "reliability": 0.84, "source": "3_malicious_domains.csv"})
    for cve in cves:
        if cve in catalogs["cves"]:
            matches.append({"type": "cve", "value": cve, "reliability": 0.86, "source": "2_cve_vulnerabilities.csv"})

    for record in catalogs["otx_records"]:
        match_score = 0
        if any(mitre in safe_text(record["attack_ids"]).upper() for mitre in mitre_list):
            match_score += 2
        if any(keyword in lowered for keyword in record["match_text"].split()[:25]):
            match_score += 1
        if match_score >= 2:
            matches.append(
                {
                    "type": "otx",
                    "value": safe_text(record["title"])[:120],
                    "reliability": 0.73,
                    "source": "1_otx_threat_intel.csv",
                    "attack_ids": safe_text(record["attack_ids"]),
                }
            )
            if len(matches) >= 5:
                break

    return matches[:8]


def build_evidence_items(row: dict[str, Any], catalogs: dict[str, Any]) -> dict[str, Any]:
    event_text = safe_text(row.get("event_text"))
    raw = row.get("raw_fields") or {}
    source_name = Path(safe_text(row.get("source_file", "unknown"))).name.lower()
    source_reliability = SOURCE_RELIABILITY.get(source_name, 0.70)
    mitre_list = extract_mitre_list(safe_text(row.get("mitre_techniques")))
    rule_signals = rule_signal_summary(event_text)
    cti_matches = match_cti(row, catalogs)
    cti_match_count = len(cti_matches)

    evidence_items: list[dict[str, Any]] = []
    for key in ("Category", "IncidentGrade", "EntityType", "EvidenceRole", "IpAddress", "Url", "DeviceName", "AccountName", "FileName"):
        value = raw.get(key)
        if value is None or safe_text(value) == "":
            continue
        evidence_items.append(
            {
                "type": key.lower(),
                "value": safe_text(value),
                "source_type": "raw_field",
                "reliability_weight": source_reliability,
                "evidence_role": safe_text(raw.get("EvidenceRole", "UNKNOWN")),
            }
        )

    for mitre in mitre_list:
        evidence_items.append(
            {
                "type": "mitre",
                "value": mitre,
                "source_type": "mitre",
                "reliability_weight": 0.80,
                "evidence_role": "attack-technique",
            }
        )

    for match in cti_matches:
        evidence_items.append(
            {
                "type": match["type"],
                "value": match["value"],
                "source_type": "cti",
                "reliability_weight": match["reliability"],
                "evidence_role": "threat-intelligence",
            }
        )

    evidence_lines = [f"{item['type']}={item['value']} (rel={item['reliability_weight']:.2f})" for item in evidence_items]
    evidence_text = " | ".join(evidence_lines[:25]) if evidence_lines else event_text[:500]

    contradiction_score = derive_contradiction_score(row, rule_signals, cti_match_count)
    evidence_diversity = len({item["type"] for item in evidence_items})
    evidence_consistency = float(np.clip(1.0 - contradiction_score, 0.0, 1.0))
    adversarial_noise_score = float(
        np.clip(
            contradiction_score
            + (0.15 if "injected_adversarial_signal=" in event_text.lower() else 0.0)
            + (0.10 if cti_match_count == 0 and safe_text(row.get("label")) == "adversarial" else 0.0),
            0.0,
            1.0,
        )
    )
    reliability_values = [item["reliability_weight"] for item in evidence_items] or [source_reliability]
    reliability_score = float(np.mean(reliability_values))
    mitre_risk_score = float(min(1.0, 0.25 * len(mitre_list) + 0.1 * sum(1 for x in mitre_list if "." in x)))
    cti_match_score = float(min(1.0, 0.20 * cti_match_count))

    return {
        "evidence_text": evidence_text,
        "evidence_items": evidence_items,
        "cti_matches": cti_matches,
        "cti_match_count": cti_match_count,
        "cti_match_score": cti_match_score,
        "mitre_list": mitre_list,
        "mitre_count": len(mitre_list),
        "mitre_risk_score": mitre_risk_score,
        "rule_signals": rule_signals,
        "contradiction_score": contradiction_score,
        "evidence_diversity": evidence_diversity,
        "evidence_consistency": evidence_consistency,
        "adversarial_noise_score": adversarial_noise_score,
        "reliability_score": reliability_score,
    }


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
