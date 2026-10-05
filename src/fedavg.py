"""Weighted FedAvg aggregation for client model state dicts."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import torch


def fed_avg(
    state_dicts: Sequence[OrderedDict[str, torch.Tensor]],
    sample_counts: Sequence[int],
) -> OrderedDict[str, torch.Tensor]:
    """Aggregate client weights: w_global = sum_i (n_i / n) * w_i."""
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have the same length")
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    total = sum(sample_counts)
    if total <= 0:
        raise ValueError("sum(sample_counts) must be positive")

    keys = state_dicts[0].keys()
    aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        weighted = None
        for state, count in zip(state_dicts, sample_counts, strict=True):
            term = state[key].float() * (count / total)
            weighted = term if weighted is None else weighted + term
        aggregated[key] = weighted
    return aggregated
