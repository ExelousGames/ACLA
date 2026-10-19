"""A standalone image/context regressor trained from scratch for signed XYZ offsets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .position_data import OUTPUT_NAMES, encode_position_context, image_tensor, reference_inputs


class PositionModel(nn.Module):
    """Learn image features and six unconstrained signed outputs together.

    Input: RGB images, [reference XYZ, centerline XYZ], normalized lap position,
    and one-hot game/map names. Output: the same two XYZ triples minus the
    driver's position, in unchanged source coordinates.
    The driver's position is deliberately absent from every inference API.
    """

    def __init__(self, imgsz=768, *, categories):
        super().__init__()
        if imgsz < 32 or imgsz % 32:
            raise ValueError("Image size must be a positive multiple of 32.")
        self.imgsz = imgsz
        self.categories = {field: list(categories[field]) for field in ("game", "map_name")}
        for field, names in self.categories.items():
            if (
                not names or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"{field} categories must contain unique, nonempty names.")
        self.context_size = 1 + sum(len(names) for names in self.categories.values())
        # Strided convolutions learn visual features directly from position targets.
        # GroupNorm supports the small batches used for full-frame images.
        layers = []
        channels = 3
        for width in (16, 32, 64, 96, 128):
            layers.extend((
                nn.Conv2d(channels, width, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, width), nn.ReLU(),
            ))
            channels = width
        self.image_encoder = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.register_buffer("position_mean", torch.zeros(6))
        self.register_buffer("position_std", torch.ones(6))
        self.position_encoder = nn.Sequential(nn.Linear(channels * 4 * 4, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 6 + self.context_size, 128), nn.ReLU(), nn.Linear(128, 6))

    def visual_features(self, images):
        return self.spatial_pool(self.image_encoder(images)).flatten(1)

    def forward(self, images, references, context):
        if references.shape != (images.shape[0], 2, 3):
            raise ValueError("References must have shape (batch, 2, 3): reference XYZ, centerline XYZ.")
        if not torch.isfinite(references).all():
            raise ValueError("Reference coordinates must be finite.")
        if context.shape != (images.shape[0], self.context_size):
            raise ValueError("Context must contain normalized position and the saved game/map one-hot vectors.")
        if not torch.isfinite(context).all() or torch.any((context < 0) | (context > 1)):
            raise ValueError("Position context values must be finite and between 0 and 1.")
        positions = (references.flatten(1) - self.position_mean) / self.position_std
        visual = self.position_encoder(self.visual_features(images))
        return self.head(torch.cat((visual, positions, context), dim=1)).reshape(-1, 2, 3)

    @torch.no_grad()
    def predict_position(self, image: Path, normalized_position, reference_lap, centerline, *, game, map_name):
        self.eval()
        device = next(self.parameters()).device
        context = torch.from_numpy(encode_position_context(
            normalized_position, game, map_name, self.categories,
        )).unsqueeze(0).to(device)
        references = torch.as_tensor(
            reference_inputs(normalized_position, reference_lap, centerline), dtype=torch.float32, device=device,
        ).unsqueeze(0)
        prediction = self(image_tensor(image, self.imgsz).unsqueeze(0).to(device), references, context)[0].cpu().tolist()
        return {"reference_delta": prediction[0], "centerline_delta": prediction[1]}

    def save(self, path: Path, *, metrics=None):
        torch.save({
            "format": "acla-position-cnn-v1",
            "categories": deepcopy(self.categories),
            "imgsz": self.imgsz,
            "state_dict": self.state_dict(),
            "output_names": list(OUTPUT_NAMES),
            "target_definition": "reference_position - current_position",
            "coordinate_units": "source_coordinates",
            "metrics": metrics or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        # Format/version labels are informational; check the actual architecture.
        if "categories" not in checkpoint:
            raise ValueError("Position checkpoint is missing game/map context; retrain older models with game/map context.")
        if "image_encoder.0.weight" not in checkpoint["state_dict"]:
            raise ValueError("Position checkpoint is missing the standalone image encoder; retrain from scratch with train-position.")
        if "position_encoder.0.weight" not in checkpoint["state_dict"]:
            raise ValueError("Position checkpoint is missing the spatial encoder; retrain from scratch with train-position.")
        model = cls(checkpoint["imgsz"], categories=checkpoint["categories"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()
