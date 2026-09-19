import os, sys
import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import run_inference_and_export


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 20, 1)
        self.mc_dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        return self.conv(self.mc_dropout(x))

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()


def run(tmp_path, ids, unc):
    return run_inference_and_export(
        TinyModel().eval(), torch.randn(2, 3, 8, 8), ids,
        str(tmp_path / "out"), with_uncertainty=unc, num_passes=3)


def test_no_uncertainty_one_pair_per_sample(tmp_path):
    res = run(tmp_path, ["a", "b"], False)
    assert sorted(os.listdir(tmp_path / "out")) == ["a_seg.png", "b_seg.png"]
    assert all(r["uncertainty_path"] is None for r in res)


def test_no_stray_dirs_beside_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(tmp_path, ["a", "b"], False)
    assert not (tmp_path / "a").exists()


def test_uncertainty_matches_input_size(tmp_path):
    res = run(tmp_path, ["a", "b"], True)
    assert len(os.listdir(tmp_path / "out")) == 4
    assert np.load(res[0]["uncertainty_path"]).shape == (8, 8)


def test_wrong_number_of_ids_raises(tmp_path):
    with pytest.raises(AssertionError):
        run(tmp_path, ["only_one"], False)
