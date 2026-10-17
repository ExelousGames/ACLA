"""Export the training model to the stable Track Vision ONNX interface."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from .position_data import OUTPUT_NAMES
from .position_model import YOLOPositionModel


METADATA_KEY = "acla.track-position"
OPSET_VERSION = 17


class _ExportSpatialPool(nn.Module):
    """Lower adaptive pooling to slices when ONNX AveragePool cannot express it."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, features):
        height, width = (int(size) for size in features.shape[-2:])
        rows, columns = self.output_size
        if height % rows == 0 and width % columns == 0:
            return nn.functional.avg_pool2d(features, (height // rows, width // columns))
        # Adaptive pooling uses floor(start) and ceil(end), including overlapping
        # bins for small/uneven grids. The exported image resolution is fixed.
        return torch.cat([
            torch.cat([
                features[
                    ..., row * height // rows:((row + 1) * height + rows - 1) // rows,
                    column * width // columns:((column + 1) * width + columns - 1) // columns,
                ].mean(dim=(-2, -1), keepdim=True)
                for column in range(columns)
            ], dim=-1)
            for row in range(rows)
        ], dim=-2)


class _TrackVisionModel(nn.Module):
    def __init__(self, position):
        super().__init__()
        self.position = position
        position.spatial_pool = _ExportSpatialPool(position.spatial_pool.output_size)
        # Keep raw one-to-many masks, including for YOLO end-to-end heads.
        segment = position.yolo.model[-1]
        segment.export, segment.format, segment.dynamic, segment.end2end = True, "onnx", False, False

    def forward(self, images, references, context):
        deltas = self.position(images, references, context)
        predictions, prototypes = self.position.yolo(images)
        return deltas, predictions, prototypes


def export_position_checkpoint(checkpoint: Path) -> Path:
    """Write a self-contained ONNX file and a matching JSON manifest beside .pt."""
    import onnx

    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("Position checkpoint must use the .pt extension")
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f"Checkpoint is missing or empty: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("output_names") != list(OUTPUT_NAMES) or not saved.get("state_dict"):
        raise ValueError("Position checkpoint is missing its weights or signed XYZ outputs.")
    if saved.get("target_definition") != "reference_position - current_position":
        raise ValueError("Position target_definition must be reference_position - current_position.")
    if saved.get("coordinate_units") != "source_coordinates":
        raise ValueError("Position coordinate_units must be source_coordinates.")
    position = YOLOPositionModel.load(checkpoint).float().eval().requires_grad_(False)
    if not torch.isfinite(position.position_mean).all() or not torch.isfinite(position.position_std).all() or not (position.position_std > 0).all():
        raise ValueError("Invalid position input standardization.")
    labels = [position.yolo.names[index] for index in range(len(position.yolo.names))]
    metadata = {
        "task": "position",
        "classNames": labels,
        "metadata": {
            "format": saved.get("format"),
            "artifactFormat": "onnx",
            "trainingRunId": checkpoint.parent.name,
            "imgsz": position.imgsz,
            "categories": position.categories,
            "outputNames": list(OUTPUT_NAMES),
            "targetDefinition": saved["target_definition"],
            "coordinateUnits": saved["coordinate_units"],
            "metrics": saved.get("metrics", {}),
        },
    }
    json.dumps(metadata, allow_nan=False)
    del saved
    inputs = (
        torch.zeros(1, 3, position.imgsz, position.imgsz),
        torch.zeros(1, 2, 3),
        torch.zeros(1, position.context_size),
    )
    model = _TrackVisionModel(position).eval()
    output = checkpoint.with_suffix(".onnx")
    with TemporaryDirectory(prefix=".position-export-", dir=checkpoint.parent) as folder:
        temporary = Path(folder) / output.name
        # The NVIDIA requirements still allow Torch 2.0, before the dynamo flag.
        export_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(
            model, inputs, str(temporary),
            input_names=["images", "references", "context"],
            output_names=["deltas", "predictions", "prototypes"],
            opset_version=OPSET_VERSION, dynamic_axes=None, **export_options,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)

        def tensors(values):
            return {
                value.name: {"dtype": "float32", "shape": [dim.dim_value for dim in value.type.tensor_type.shape.dim]}
                for value in values
            }

        input_specs, output_specs = tensors(graph.graph.input), tensors(graph.graph.output)
        shapes = [spec["shape"] for spec in output_specs.values()]
        if (
            [spec["shape"] for spec in input_specs.values()] != [list(value.shape) for value in inputs]
            or len(shapes) != 3 or shapes[0] != [1, 2, 3]
            or len(shapes[1]) != 3 or len(shapes[2]) != 4
            or shapes[1][0] != 1 or shapes[2][0] != 1
            or shapes[1][1] != 4 + len(labels) + shapes[2][1]
        ):
            raise ValueError("Unexpected Track Vision ONNX inputs or outputs.")
        metadata["metadata"]["inference"] = {
            "schemaVersion": 1,
            "opset": OPSET_VERSION,
            "inputs": input_specs,
            "outputs": output_specs,
            "image": {
                "colorSpace": "RGB", "layout": "NCHW", "scale": 1 / 255,
                "resize": "letterbox", "paddingValue": 114,
                "sizeRounding": "ties-to-even", "paddingPlacement": "center-floor",
            },
            "referenceOrder": ["reference", "centerline"],
            "coordinateOrder": ["x", "y", "z"],
            "contextOrder": ["normalized_position", "game", "map_name"],
            "categoryEncoding": "one-hot",
        }
        manifest = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        onnx.helper.set_model_props(graph, {METADATA_KEY: manifest})
        onnx.save_model(graph, str(temporary), save_as_external_data=False)
        temporary.replace(output)
    output.with_suffix(".onnx.json").write_text(manifest, encoding="utf-8")
    return output
