"""
Module 6 — Rule Version Loader & Validator (drift/rules_loader.py)

Loads, validates, and manages versioned rule-weight configurations
(Visa CE3.0 and Mastercard First-Party Trust revisions).

All rule files are clearly labeled as "approximated for demonstration"
per docs/DATASETS_GUIDE.md Section D.
"""

from typing import Dict, List, Any, Optional, Union
import os
from pathlib import Path
from dataclasses import dataclass, field
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RULES_DIR = REPO_ROOT / "rules"

REQUIRED_CRITERIA = [
    "prior_undisputed_in_window",
    "device_match",
    "shipping_match",
    "ip_match",
    "cvv_avs_verified",
    "otp_3ds_authenticated",
]


@dataclass
class RuleVersionConfig:
    """Structured representation of a versioned evidentiary rule set."""
    version: str
    effective_date: str
    network: str
    name: str
    description: str
    status: str
    criteria_weights: Dict[str, float]
    decision_threshold: float = 0.55
    noise_sigma: float = 0.08
    non_ce3_penalty_factor: float = 0.40
    raw_path: Optional[str] = None

    def __post_init__(self):
        # Normalize weights to ensure sum == 1.0
        total = sum(self.criteria_weights.values())
        if total > 0:
            self.criteria_weights = {k: round(v / total, 4) for k, v in self.criteria_weights.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "effective_date": self.effective_date,
            "network": self.network,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "criteria_weights": self.criteria_weights,
            "decision_threshold": self.decision_threshold,
            "noise_sigma": self.noise_sigma,
            "non_ce3_penalty_factor": self.non_ce3_penalty_factor,
            "raw_path": str(self.raw_path) if self.raw_path else None,
        }


def list_available_rule_versions(rules_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Returns a list of available rule version names in the specified directory."""
    r_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    if not r_dir.exists():
        return []
    versions = []
    for p in sorted(r_dir.glob("*.yaml")):
        versions.append(p.stem)
    return versions


def load_rule_version(
    version_or_path: str,
    rules_dir: Optional[Union[str, Path]] = None,
) -> RuleVersionConfig:
    """
    Loads and validates a rule configuration file.
    Supports either version alias (e.g., 'ce3_2023', 'ce3_2026_04') or explicit file path.
    """
    r_dir = Path(rules_dir) if rules_dir else DEFAULT_RULES_DIR
    
    if os.path.exists(version_or_path):
        target_path = Path(version_or_path)
    else:
        # Check in rules directory with or without .yaml
        target_path = r_dir / (version_or_path if version_or_path.endswith(".yaml") else f"{version_or_path}.yaml")
        if not target_path.exists():
            # Check if nested in drift/rules
            nested_path = REPO_ROOT / "drift" / "rules" / (version_or_path if version_or_path.endswith(".yaml") else f"{version_or_path}.yaml")
            if nested_path.exists():
                target_path = nested_path
            else:
                available = list_available_rule_versions(r_dir)
                raise FileNotFoundError(
                    f"Rule version file not found: '{version_or_path}'. Available versions: {available}"
                )

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {target_path}: expected dictionary root.")

    weights = data.get("criteria_weights", {})
    for req in REQUIRED_CRITERIA:
        if req not in weights:
            raise ValueError(f"Rule version '{target_path.stem}' missing required criterion: '{req}'")

    config = RuleVersionConfig(
        version=str(data.get("version", target_path.stem)),
        effective_date=str(data.get("effective_date", "unknown")),
        network=str(data.get("network", "Visa")),
        name=str(data.get("name", target_path.stem)),
        description=str(data.get("description", "")),
        status=str(data.get("status", "approximated_for_demonstration")),
        criteria_weights={k: float(v) for k, v in weights.items()},
        decision_threshold=float(data.get("decision_threshold", 0.55)),
        noise_sigma=float(data.get("noise_sigma", 0.08)),
        non_ce3_penalty_factor=float(data.get("non_ce3_penalty_factor", 0.40)),
        raw_path=str(target_path),
    )

    return config


def get_rule_weights(
    version_or_path: str,
    rules_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Convenience helper returning just normalized criteria weights dictionary."""
    cfg = load_rule_version(version_or_path, rules_dir=rules_dir)
    return cfg.criteria_weights
