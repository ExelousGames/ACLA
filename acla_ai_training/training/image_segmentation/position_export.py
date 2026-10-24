"""Export the standalone regressor to the position-only Track Vision ONNX interface."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from .position_data import COORDINATE_AXES, COORDINATE_FRAME, OUTPUT_NAMES, TARGET_DEFINITION
from .position_model import PositionModel


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


def export_position_checkpoint(checkpoint: Path) -> Path:
    """Write a self-contained ONNX file and a matching JSON manifest beside .pt."""
    import onnx

    checkpoint = checkpoint.resolve()
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("Position checkpoint must use the .pt extension")
    if not checkpoint.is_file() or not checkpoint.stat().st_size:
        raise ValueError(f"Checkpoint is missing or empty: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not saved.get("state_dict"):
        raise ValueError("Position checkpoint is missing its weights.")
    if saved.get("output_names") != list(OUTPUT_NAMES):
        raise ValueError("Position checkpoint must predict only reference-lap XYZ deltas; retrain from scratch with train-position.")
    if (saved.get("target_definition") != TARGET_DEFINITION
            or saved.get("coordinate_frame") != COORDINATE_FRAME
            or saved.get("coordinate_axes") != list(COORDINATE_AXES)):
        raise ValueError("Position targets must be car-local right/forward/up reference offsets.")
    if saved.get("coordinate_units") != "source_coordinates":
        raise ValueError("Position coordinate_units must be source_coordinates.")
    position = PositionModel.load(checkpoint).float().eval().requires_grad_(False)
    if not torch.isfinite(position.position_mean).all() or not torch.isfinite(position.position_std).all() or not (position.position_std > 0).all():
        raise ValueError("Invalid position input standardization.")
    metadata = {
        "task": "position",
        # The shared model store requires names; these identify regression axes.
        "classNames": list(OUTPUT_NAMES),
        "metadata": {
            "format": saved.get("format"),
            "artifactFormat": "onnx",
            "trainingRunId": checkpoint.parent.name,
            "imgsz": position.imgsz,
            "categories": position.categories,
            "outputNames": list(OUTPUT_NAMES),
            "targetDefinition": saved["target_definition"],
            "coordinateUnits": saved["coordinate_units"],
            "coordinateFrame": saved["coordinate_frame"],
            "coordinateAxes": saved["coordinate_axes"],
            "metrics": saved.get("metrics", {}),
        },
    }
    json.dumps(metadata, allow_nan=False)
    del saved
    inputs = (
        torch.zeros(1, 3, position.imgsz, position.imgsz),
        torch.zeros(1, 1, 3),
        torch.zeros(1, position.context_size),
    )
    temporal = position.sequence_length > 1
    if temporal:
        inputs = tuple(value.unsqueeze(1).repeat(1, position.sequence_length, *([1] * (value.ndim - 1))) for value in inputs)
    position.spatial_pool = _ExportSpatialPool(position.spatial_pool.output_size)
    output = checkpoint.with_suffix(".onnx")
    with TemporaryDirectory(prefix=".position-export-", dir=checkpoint.parent) as folder:
        temporary = Path(folder) / output.name
        # The NVIDIA requirements still allow Torch 2.0, before the dynamo flag.
        export_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(
            position, inputs, str(temporary),
            input_names=["images", "references", "context"],
            output_names=["deltas"],
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
        if (
            [spec["shape"] for spec in input_specs.values()] != [list(value.shape) for value in inputs]
            or output_specs != {"deltas": {"dtype": "float32", "shape": [1, 1, 3]}}
        ):
            raise ValueError("Unexpected Track Vision ONNX inputs or outputs.")
        metadata["metadata"]["inference"] = {
            "schemaVersion": 5 if temporal else 4,
            "opset": OPSET_VERSION,
            "inputs": input_specs,
            "outputs": output_specs,
            "image": {
                "colorSpace": "RGB", "layout": "NTCHW" if temporal else "NCHW", "scale": 1 / 255,
                "resize": "letterbox", "paddingValue": 114,
                "sizeRounding": "ties-to-even", "paddingPlacement": "center-floor",
            },
            "referenceOrder": ["reference"],
            "coordinateOrder": ["x", "y", "z"],
            "contextOrder": ["normalized_position", "game", "map_name"],
            "categoryEncoding": "one-hot",
        }
        if temporal:
            metadata["metadata"]["inference"]["temporal"] = {
                "type": "gru", "sequenceLength": position.sequence_length,
                "order": "oldest-to-newest", "targetFrame": "last",
                "padding": "repeat-first", "state": "reset-per-window",
                "resetHistoryOn": ["session-change", "recording-change", "game-change", "map-change", "seek"],
            }
        manifest = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        onnx.helper.set_model_props(graph, {METADATA_KEY: manifest})
        onnx.save_model(graph, str(temporary), save_as_external_data=False)
        temporary.replace(output)
    output.with_suffix(".onnx.json").write_text(manifest, encoding="utf-8")
    return output
