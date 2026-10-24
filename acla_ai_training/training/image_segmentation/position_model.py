"""A standalone image/context regressor trained from scratch for signed XYZ offsets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .position_data import COORDINATE_AXES, COORDINATE_FRAME, OUTPUT_NAMES, TARGET_DEFINITION, encode_position_context, image_tensor


class PositionModel(nn.Module):
    """Learn image features and three unconstrained signed outputs together.

    Input: RGB images, reference-lap XYZ, normalized lap position,
    and one-hot game/map names. Output: the offset from the driver to the
    reference lap in car-local XYZ (right, forward, up), in source units.
    With sequence_length > 1, inputs include an oldest-first time dimension;
    the GRU summarizes that window and the output describes its final frame.
    The driver's position is deliberately absent from every inference API.
    """

    def __init__(self, imgsz=768, *, categories, sequence_length=1):
        super().__init__()
        if imgsz < 32 or imgsz % 32:
            raise ValueError("Image size must be a positive multiple of 32.")
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("Sequence length must be a positive integer.")
        self.imgsz = imgsz
        self.sequence_length = sequence_length
        self.categories = {field: list(categories[field]) for field in ("game", "map_name")}
        for field, names in self.categories.items():
            if (
                not names or any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"{field} categories must contain unique, nonempty names.")
        self.context_size = 1 + sum(len(names) for names in self.categories.values())
        # Convolutions learn visual features directly from position targets.
        # GroupNorm supports the small batches used for full-frame images.
        layers = []
        channels = 3
        strides = (2, 2, 2, 1, 1)
        for width, stride in zip((16, 32, 64, 96, 128), strides):
            layers.extend((
                nn.Conv2d(channels, width, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(8, width), nn.ReLU(),
            ))
            channels = width
        self.image_encoder = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.register_buffer("position_mean", torch.zeros(3))
        self.register_buffer("position_std", torch.ones(3))
        self.position_encoder = nn.Sequential(nn.Linear(channels * 4 * 4, 128), nn.ReLU())
        self.temporal_encoder = nn.GRUCell(128 + 3 + self.context_size, 128) if sequence_length > 1 else None
        self.head = nn.Sequential(nn.Linear(128 + 3 + self.context_size, 128), nn.ReLU(), nn.Linear(128, 3))

    def visual_features(self, images):
        return self.spatial_pool(self.image_encoder(images)).flatten(1)

    def forward(self, images, references, context):
        batch = images.shape[0]
        if self.sequence_length > 1:
            if images.shape[1:] != (self.sequence_length, 3, self.imgsz, self.imgsz):
                raise ValueError("Temporal images must have shape (batch, sequence_length, 3, imgsz, imgsz).")
            if references.shape != (batch, self.sequence_length, 1, 3):
                raise ValueError("Temporal references must have shape (batch, sequence_length, 1, 3).")
            if context.shape != (batch, self.sequence_length, self.context_size):
                raise ValueError("Temporal context must have shape (batch, sequence_length, context_size).")
            images = images.flatten(0, 1)
            references = references.flatten(0, 1)
            context = context.flatten(0, 1)
        if references.shape != (images.shape[0], 1, 3):
            raise ValueError("References must have shape (batch, 1, 3): reference-lap XYZ.")
        if not torch.isfinite(references).all():
            raise ValueError("Reference coordinates must be finite.")
        if context.shape != (images.shape[0], self.context_size):
            raise ValueError("Context must contain normalized position and the saved game/map one-hot vectors.")
        if not torch.isfinite(context).all() or torch.any((context < 0) | (context > 1)):
            raise ValueError("Position context values must be finite and between 0 and 1.")
        positions = (references.flatten(1) - self.position_mean) / self.position_std
        visual = self.position_encoder(self.visual_features(images))
        if self.temporal_encoder is not None:
            features = torch.cat((visual, positions, context), dim=1).reshape(batch, self.sequence_length, -1)
            # Explicitly unroll the causal GRU so ONNX uses ordinary tensor ops.
            # Each window starts from zero; no state can leak between batches.
            visual = visual.new_zeros(batch, 128)
            for step in features.unbind(1):
                visual = self.temporal_encoder(step, visual)
            positions = positions.reshape(batch, self.sequence_length, 3)[:, -1]
            context = context.reshape(batch, self.sequence_length, self.context_size)[:, -1]
        return self.head(torch.cat((visual, positions, context), dim=1)).reshape(-1, 1, 3)

    @torch.no_grad()
    def predict_position(self, image: Path, normalized_position, reference_lap, *, game, map_name, history=None):
        """Predict the current frame; history is oldest-first (image, lap_position) pairs.

        Supply only preceding frames from the same recording/game/map. Missing
        history repeats the earliest supplied frame, matching training padding.
        """
        self.eval()
        device = next(self.parameters()).device
        frames = [*(history or []), (image, normalized_position)][-self.sequence_length:]
        frames = [frames[0]] * (self.sequence_length - len(frames)) + frames
        context = torch.stack([torch.from_numpy(encode_position_context(
            position, game, map_name, self.categories,
        )) for _, position in frames]).to(device)
        references = torch.stack([
            torch.as_tensor(reference_lap.at(position), dtype=torch.float32).reshape(1, 3)
            for _, position in frames
        ]).to(device)
        images = torch.stack([image_tensor(path, self.imgsz) for path, _ in frames]).to(device)
        if self.sequence_length > 1:
            images, references, context = (value.unsqueeze(0) for value in (images, references, context))
        prediction = self(images, references, context)[0].cpu().tolist()
        return {"reference_delta": prediction[0]}

    def save(self, path: Path, *, metrics=None):
        torch.save({
            "format": "acla-position-cnn-gru-v1" if self.sequence_length > 1 else "acla-position-cnn-v3",
            "sequence_length": self.sequence_length,
            "categories": deepcopy(self.categories),
            "imgsz": self.imgsz,
            "state_dict": self.state_dict(),
            "output_names": list(OUTPUT_NAMES),
            "target_definition": TARGET_DEFINITION,
            "coordinate_frame": COORDINATE_FRAME,
            "coordinate_axes": list(COORDINATE_AXES),
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
        if checkpoint.get("target_definition") != TARGET_DEFINITION:
            raise ValueError("Position checkpoint must use corrected right/forward/up targets; migrate the dataset and retrain.")
        model = cls(checkpoint["imgsz"], categories=checkpoint["categories"],
                    sequence_length=checkpoint.get("sequence_length", 1))
        if checkpoint["state_dict"]["head.0.weight"].shape != model.head[0].weight.shape:
            raise ValueError("Position checkpoint uses centerline inputs; retrain from scratch with train-position.")
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()
