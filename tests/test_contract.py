from pathlib import Path
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from policy import camera_mosaic, support_mask
from models.tactile_expert import UniversalTactileExpertConfig
from models.tactile_autoencoder.metadata import packed_metadata
from models.tactile_autoencoder.normalizer import ForceNormalizerV41
from models.tactile_autoencoder.normalization_config import ForceNormalizationV41Config


def test_layout_and_support():
    c = UniversalTactileExpertConfig()
    assert (c.condition_slices, c.future_slices, c.sequence_length) == (1, 16, 204)
    support = torch.from_numpy(support_mask())[:, None]
    assert support.flatten(1).sum(1).tolist() == [113, 112, 113, 112]
    slots = torch.tensor([[0,1,2,3],[4,5,6,7],[17,18,19,20],[21,22,23,24]])
    native, m = packed_metadata(support, slots)
    assert native.tolist() == [0, 1, 16, 17]
    assert m["hand_side_id"].tolist() == [1, 1, 2, 2]
    assert m["finger_id"].tolist() == [1, 2, 1, 2]
    assert m["segment_id"].tolist() == [2, 2, 2, 2]


def test_normalizer_zero_deadband_and_limits():
    n = ForceNormalizerV41(ForceNormalizationV41Config())
    f = torch.zeros(1, 4, 1, 3, 10, 14)
    s = torch.from_numpy(support_mask())[None, :, None, None]
    assert n.normalize(f, s)[0].count_nonzero() == 0
    f[..., 0, :, :] = 1e-5
    assert n.normalize(f, s)[0].count_nonzero() == 0
    f[..., 0, :, :] = 50
    f[..., 1, :, :] = -5
    f[..., 2, :, :] = 6.1
    norm = n.normalize(f, s)[0]
    assert norm.max() <= 1 and norm.min() >= -1
    torch.testing.assert_close(n.denormalize(norm, s), f * s)


def test_camera_color_and_padding():
    img = np.zeros((480,640,3), dtype=np.uint8)
    img[...,0] = 255
    out = camera_mosaic(img,img,img,"bgr")
    assert out.shape == (384,320,3)
    assert out[:12].sum() == 0 and out[-12:].sum() == 0
    assert out[12,0].tolist() == [0,0,1]
    with pytest.raises(ValueError):
        UniversalTactileExpertConfig(condition_slices=2)
