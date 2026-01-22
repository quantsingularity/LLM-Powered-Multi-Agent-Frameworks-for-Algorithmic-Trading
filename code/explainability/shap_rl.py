"""SHAP analysis for RL policy decisions.

For SB3 policies, we approximate action logits/values using the policy network
and use KernelExplainer as a generic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np


@dataclass
class ShapConfig:
    background_samples: int = 200
    nsamples: int = 200


def make_policy_predict_fn(sb3_model) -> Callable[[np.ndarray], np.ndarray]:
    def predict(x: np.ndarray) -> np.ndarray:
        # SB3 expects shape (n, obs_dim)
        acts = []
        for row in x:
            a, _ = sb3_model.predict(row, deterministic=True)
            acts.append(a)
        return np.array(acts).reshape(-1, 1)

    return predict


def shap_explain_actions(
    sb3_model, observations: np.ndarray, cfg: ShapConfig = ShapConfig()
) -> Dict[str, Any]:
    import shap

    predict_fn = make_policy_predict_fn(sb3_model)

    bg = observations[: min(len(observations), cfg.background_samples)]
    explainer = shap.KernelExplainer(predict_fn, bg)

    vals = explainer.shap_values(
        observations[: min(len(observations), 50)], nsamples=cfg.nsamples
    )
    return {
        "shap_values": vals,
        "expected_value": explainer.expected_value,
    }
