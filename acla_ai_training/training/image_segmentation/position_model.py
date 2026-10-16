"""A signed XYZ regression head on the existing YOLO segmentation features."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from .position_data import OUTPUT_NAMES, image_tensor, reference_inputs


class YOLOPositionModel(nn.Module):
    """Freeze segmentation weights and learn six unconstrained signed outputs.

    Input: RGB images and [reference XYZ, centerline XYZ]. Output: the same two
    XYZ triples minus the driver's position, in unchanged source coordinates.
    The driver's position is deliberately absent from every inference API.
    """

    def __init__(self, yolo, imgsz=768):
        super().__init__()
        if imgsz < 32 or imgsz % 32:
            raise ValueError("Image size must be a positive multiple of 32.")
        self.yolo = yolo.float().eval().requires_grad_(False)
        self.imgsz = imgsz
        # Use all three feature scales feeding the existing segmentation head.
        self.feature_layers = list(self.yolo.model[-1].f)
        with torch.no_grad():
            device = next(self.yolo.parameters()).device
            width = self.visual_features(torch.zeros(1, 3, 64, 64, device=device)).shape[1]
        self.register_buffer("position_mean", torch.zeros(6))
        self.register_buffer("position_std", torch.ones(6))
        self.head = nn.Sequential(nn.Linear(width + 6, 128), nn.ReLU(), nn.Linear(128, 6))

    @classmethod
    def from_segmentation_checkpoint(cls, checkpoint: Path, imgsz=768):
        from ultralytics import YOLO

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Existing segmentation checkpoint required: {checkpoint}")
        network = YOLO(str(checkpoint), task="segment")
        if network.task != "segment":
            raise ValueError("Position regression requires a YOLO segmentation checkpoint.")
        return cls(network.model, imgsz)

    def train(self, mode=True):
        super().train(mode)
        # Freeze BatchNorm statistics as well as segmentation parameters.
        self.yolo.eval()
        return self

    def visual_features(self, images):
        with torch.no_grad():
            return torch.stack(self.yolo.predict(images, embed=self.feature_layers))

    def forward(self, images, references):
        if references.shape != (images.shape[0], 2, 3):
            raise ValueError("References must have shape (batch, 2, 3): reference XYZ, centerline XYZ.")
        if not torch.isfinite(references).all():
            raise ValueError("Reference coordinates must be finite.")
        positions = (references.flatten(1) - self.position_mean) / self.position_std
        return self.head(torch.cat((self.visual_features(images), positions), dim=1)).reshape(-1, 2, 3)

    @torch.no_grad()
    def predict_position(self, image: Path, normalized_position, reference_lap, centerline):
        self.eval()
        device = next(self.parameters()).device
        references = torch.as_tensor(
            reference_inputs(normalized_position, reference_lap, centerline), dtype=torch.float32, device=device,
        ).unsqueeze(0)
        prediction = self(image_tensor(image, self.imgsz).unsqueeze(0).to(device), references)[0].cpu().tolist()
        return {"reference_delta": prediction[0], "centerline_delta": prediction[1]}

    def save(self, path: Path, *, metrics=None):
        torch.save({
            "format": "acla-yolo-position-v1",
            "imgsz": self.imgsz,
            "yolo_yaml": deepcopy(self.yolo.yaml),
            "names": self.yolo.names,
            "state_dict": self.state_dict(),
            "output_names": list(OUTPUT_NAMES),
            "target_definition": "reference_position - current_position",
            "coordinate_units": "source_coordinates",
            "metrics": metrics or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device="cpu"):
        from ultralytics.nn.tasks import SegmentationModel

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "acla-yolo-position-v1":
            raise ValueError("Not an ACLA YOLO position checkpoint.")
        yolo = SegmentationModel(deepcopy(checkpoint["yolo_yaml"]), verbose=False)
        yolo.names = checkpoint["names"]
        model = cls(yolo, checkpoint["imgsz"])
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device).eval()
