"""Prompt versioning + A/B testing.

Stores prompts in code/prompts/versions/*.yaml and experiment configs in
code/prompts/experiments/*.yaml.

At runtime, load a prompt version and optionally route traffic between
variants for A/B tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import random
import yaml


@dataclass
class PromptVersion:
    name: str
    version: str
    system_prompt: str
    user_template: str
    few_shots: list[dict]


class PromptRegistry:
    def __init__(self, base_dir: str = "code/prompts"):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.exp_dir = self.base_dir / "experiments"

    def load_version(self, name: str, version: str) -> PromptVersion:
        p = self.versions_dir / f"{name}__{version}.yaml"
        data = yaml.safe_load(p.read_text())
        return PromptVersion(
            name=name,
            version=version,
            system_prompt=data["system_prompt"],
            user_template=data["user_template"],
            few_shots=data.get("few_shots", []),
        )

    def choose_variant(
        self, experiment_name: str, rng_seed: Optional[int] = None
    ) -> Dict[str, str]:
        p = self.exp_dir / f"{experiment_name}.yaml"
        exp = yaml.safe_load(p.read_text())
        variants = exp["variants"]
        weights = [v.get("weight", 1.0) for v in variants]
        if rng_seed is not None:
            random.seed(rng_seed)
        chosen = random.choices(variants, weights=weights, k=1)[0]
        return {
            "prompt": chosen["prompt"],
            "version": chosen["version"],
            "experiment": experiment_name,
        }
