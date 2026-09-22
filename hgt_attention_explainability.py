"""
hgt_attention_explainability.py
==================================
Optional extension alongside explainability.py: surfaces HGT's OWN
internal attention weights as an additional explanation signal.
explainability.py itself needs ZERO changes for its existing
gradient/GNNExplainer methods to work on an HGTAnomalyDetector — that
was a deliberate design goal of model_hgt.py (identical forward(data)
contract to the other two models). This file is purely additive on top:
a different, HGT-specific signal, not a replacement for what already
works.

LOWER CONFIDENCE THAN THE REST OF THIS EXTENSION
─────────────────────────────────────────────────────────────────────────
HGTConv's exact mechanism for returning attention weights varies more
across PyG versions than almost anything else touched here, and cannot
be checked in this environment (no torch install — see explainability.py's
and model_hgt.py's identical caveat). This tries the most commonly
documented pattern for GAT-family layers — forward(..., return_attention_
weights=True) -> (out, (edge_index, alpha)) — with a clear, logged
failure path if the installed HGTConv doesn't support it, rather than
producing silently-wrong numbers. Verify against your installed PyG
version before using this for anything beyond a debugging aid.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from model_hgt import HGTAnomalyDetector

log = logging.getLogger(__name__)

EdgeTriple = Tuple[str, str, str]


class HGTAttentionExplainer:
    """
    Usage:
        explainer = HGTAttentionExplainer(hgt_model)
        attn = explainer.attention_for_layer(data, layer=0)
        # attn: whatever HGTConv's return_attention_weights=True hands
        # back for your installed PyG version, or None (logged) if that
        # kwarg isn't supported.
    """

    def __init__(self, model: HGTAnomalyDetector):
        self.model = model

    @torch.no_grad()
    def attention_for_layer(self, data: HeteroData, layer: int = 0):
        self.model.eval()
        encoder = self.model.encoder
        conv = encoder.convs[layer]

        x_dict = {ntype: data[ntype].x for ntype in encoder.input_proj if ntype in data.node_types}
        edge_index_dict = {t: data[t].edge_index for t in self.model.edge_types if t in data.edge_types}
        edge_attr_dict = {t: data[t].edge_attr for t in self.model.edge_types if t in data.edge_types}

        # Re-run input projection + prior layers + THIS layer's own
        # edge-bias injection exactly as HGTEncoder.forward() does, so
        # the attention weights pulled here are computed on the same
        # inputs the model actually used — not a fresh, differently
        # biased x_dict from skipping straight to `layer`.
        h_dict = {nt: F.gelu(encoder.input_proj[nt](x)) for nt, x in x_dict.items()}
        for i in range(layer):
            biased = encoder._inject_edge_bias(i, h_dict, edge_index_dict, edge_attr_dict)
            out_dict = encoder.convs[i](biased, edge_index_dict)
            h_dict = {
                nt: encoder.dropout(encoder.norms[i][nt](h_dict[nt] + out_dict.get(nt, torch.zeros_like(h_dict[nt]))))
                for nt in h_dict
            }
        biased = encoder._inject_edge_bias(layer, h_dict, edge_index_dict, edge_attr_dict)

        try:
            _, attn = conv(biased, edge_index_dict, return_attention_weights=True)
            return attn
        except TypeError as exc:
            log.warning(
                "This installed PyG version's HGTConv does not accept "
                "return_attention_weights=True (%s). No attention signal "
                "available from this path — explainability.py's gradient-"
                "based EdgeExplainer still works on this model unmodified; "
                "use that instead. See this module's honest caveat.",
                exc,
            )
            return None
