"""Unit tests for FedAvg aggregation."""

from collections import OrderedDict

import torch

from src.fedavg import fed_avg


def test_fed_avg_two_clients_equal_weight():
    w1 = OrderedDict({"layer": torch.tensor([1.0, 3.0])})
    w2 = OrderedDict({"layer": torch.tensor([3.0, 5.0])})
    out = fed_avg([w1, w2], sample_counts=[50, 50])
    assert torch.allclose(out["layer"], torch.tensor([2.0, 4.0]))


def test_fed_avg_weighted_by_sample_count():
    w1 = OrderedDict({"layer": torch.tensor([0.0])})
    w2 = OrderedDict({"layer": torch.tensor([10.0])})
    out = fed_avg([w1, w2], sample_counts=[25, 75])
    assert torch.allclose(out["layer"], torch.tensor([7.5]))
