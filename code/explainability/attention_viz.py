"""Attention visualization for local transformer LLMs.

If using HF local models with output_attentions=True, this helper converts
attention tensors to a lightweight JSON format that can be rendered in a
notebook or simple HTML report.
"""

from __future__ import annotations

from typing import Any, Dict


def attentions_to_json(attentions, tokens) -> Dict[str, Any]:
    # attentions: tuple(layers) each (batch, heads, seq, seq)
    out = {"tokens": tokens, "layers": []}
    for li, layer in enumerate(attentions):
        # take batch 0
        a = layer[0].detach().cpu().numpy()
        # average heads
        a_mean = a.mean(axis=0).tolist()
        out["layers"].append({"layer": li, "attn": a_mean})
    return out
