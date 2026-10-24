import math

import numpy as np
import pytest
import torch
from PIL import Image

from training.image_segmentation import position_data
from training.image_segmentation.position_data import PositionDataset, hud_coarse_dropout, image_tensor


@pytest.mark.parametrize("size", [(1252, 720), (626, 360), (1878, 1080)])
@pytest.mark.parametrize("noise", [False, True])
def test_dropout_covers_complete_scaled_panels_only(monkeypatch, size, noise):
    annotations = (
        (2, 1, 1245, 15), (20, 41, 211, 98), (547, 19, 170, 21),
        (1186, 38, 38, 109), (23, 652, 262, 49), (441, 435, 434, 238),
        (849, 523, 147, 188), (999, 618, 60, 90), (1057, 551, 194, 168),
    )
    regions = position_data.HUD_REGIONS
    assert len(regions) == len(annotations)
    source = Image.new("RGB", size, (17, 34, 51))
    original = np.array(source)
    for region, (x, y, width, height) in zip(regions, annotations):
        # Isolate each panel and force augmentation plus the chosen fill mode.
        monkeypatch.setattr(position_data, "HUD_REGIONS", (region,))
        decisions = iter((0.0, 0.75 if noise else 0.0))
        monkeypatch.setattr(torch, "rand", lambda *args: torch.tensor(next(decisions)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            result = np.array(hud_coarse_dropout(source))
        left, top = math.floor(x * size[0] / 1252), math.floor(y * size[1] / 720)
        right = math.ceil((x + width) * size[0] / 1252)
        bottom = math.ceil((y + height) * size[1] / 720)
        covered = np.zeros((size[1], size[0]), dtype=bool)
        covered[top:bottom, left:right] = True
        np.testing.assert_array_equal(result[~covered], original[~covered])
        patch = result[covered]
        assert np.any(patch != original[covered], axis=1).mean() > 0.99
        if noise:
            assert np.unique(patch, axis=0).shape[0] > 1
        else:
            assert (patch == patch[0]).all()
        np.testing.assert_array_equal(source, original)
        assert result.shape == original.shape and result.dtype == np.uint8


def test_training_reloads_mix_originals_and_variants_without_changing_labels(tmp_path):
    path = tmp_path / "frame.png"
    Image.new("RGB", (626, 360), (17, 34, 51)).save(path)
    original_file = path.read_bytes()
    record = {
        "image": path, "game": "acc", "map_name": "Monza", "normalized_position": 0.25,
        "references": np.array([[1, 2, 3]], dtype=np.float32),
        "target": np.array([[0.1, -0.2, 0.3]], dtype=np.float32),
    }
    categories = {"game": ["acc"], "map_name": ["Monza"]}
    validation = PositionDataset([record], 64, categories)
    training = PositionDataset([record], 64, categories, augment=True)
    original = validation[0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        samples = [training[0] for _ in range(16)]
        torch.manual_seed(7)
        for sample in samples:
            torch.testing.assert_close(training[0]["image"], sample["image"], rtol=0, atol=0)
    variants = [sample["image"] for sample in samples if not torch.equal(sample["image"], original["image"])]
    assert 1 < len(variants) < len(samples)
    assert any(not torch.equal(variants[0], image) for image in variants[1:])
    for sample in samples:
        for field in ("references", "context", "target"):
            torch.testing.assert_close(sample[field], original[field], rtol=0, atol=0)
        for field in ("game", "map_name", "normalized_position"):
            assert sample[field] == original[field]
    torch.testing.assert_close(validation[0]["image"], original["image"], rtol=0, atol=0)
    assert path.read_bytes() == original_file


def test_dropout_runs_before_letterboxing_and_is_disabled_by_default(tmp_path, monkeypatch):
    path = tmp_path / "frame.png"
    Image.new("RGB", (80, 40), "red").save(path)
    calls = []

    def obscure(image):
        calls.append((image.size, image.mode))
        return Image.new("RGB", image.size, "blue")

    monkeypatch.setattr(position_data, "hud_coarse_dropout", obscure)
    original = image_tensor(path, 64)
    assert calls == []
    augmented = image_tensor(path, 64, augment=True)
    assert calls == [((80, 40), "RGB")]
    torch.testing.assert_close(augmented[:, :16], original[:, :16])
    torch.testing.assert_close(augmented[:, 48:], original[:, 48:])
    torch.testing.assert_close(augmented[:, 16:48], torch.tensor([0, 0, 1.0])[:, None, None].expand(3, 32, 64))
