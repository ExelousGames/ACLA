"""Compare position-only ONNX outputs with the standalone regression model."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from training.image_segmentation.position_export import METADATA_KEY, export_position_checkpoint
from training.image_segmentation.position_model import PositionModel


@pytest.mark.parametrize("legacy_export_api", [False, True])
@pytest.mark.parametrize("imgsz", [32, 64, 96, 128])
def test_export_preserves_position_without_segmentation_with_small_uneven_and_even_grids(
    tmp_path, monkeypatch, legacy_export_api, imgsz,
):
    if legacy_export_api:
        export = torch.onnx.export

        def legacy_export(model, inputs, output, *, input_names, output_names, opset_version, dynamic_axes):
            return export(model, inputs, output, input_names=input_names, output_names=output_names,
                          opset_version=opset_version, dynamic_axes=dynamic_axes, dynamo=False)

        monkeypatch.setattr(torch.onnx, "export", legacy_export)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(3)
        categories = {"game": ["acc"], "map_name": ["monza", "spa"]}
        model = PositionModel(imgsz=imgsz, categories=categories).eval()
        model.position_mean.copy_(torch.arange(3))
        model.position_std.copy_(torch.arange(1, 4))
        checkpoint = tmp_path / "best.pt"
        model.save(checkpoint, metrics={"val_mae": 0.25})
        original = checkpoint.read_bytes()
        output = export_position_checkpoint(checkpoint)
        assert checkpoint.read_bytes() == original
        assert output == checkpoint.with_suffix(".onnx")
        graph = onnx.load(str(output))
        assert all(not tensor.external_data for tensor in graph.graph.initializer)
        assert sum(node.op_type == "Conv" and node.input[0] == "images" for node in graph.graph.node) == 1
        manifest = json.loads(output.with_suffix(".onnx.json").read_text())
        assert json.loads({item.key: item.value for item in graph.metadata_props}[METADATA_KEY]) == manifest
        assert manifest["task"] == "position"
        assert manifest["classNames"] == manifest["metadata"]["outputNames"]
        metadata = manifest["metadata"]
        assert metadata["outputNames"] == ["reference_delta_x", "reference_delta_y", "reference_delta_z"]
        assert metadata["artifactFormat"] == "onnx"
        assert metadata["categories"] == categories
        assert metadata["imgsz"] == imgsz
        assert metadata["metrics"] == {"val_mae": 0.25}
        contract = metadata["inference"]
        assert contract["inputs"] == {
            "images": {"dtype": "float32", "shape": [1, 3, imgsz, imgsz]},
            "references": {"dtype": "float32", "shape": [1, 1, 3]},
            "context": {"dtype": "float32", "shape": [1, 4]},
        }
        assert contract["schemaVersion"] == 4
        assert metadata["targetDefinition"] == "inverse_world_rotation @ (reference_position - current_position)"
        assert metadata["coordinateFrame"] == "car_local"
        assert metadata["coordinateAxes"] == ["right", "forward", "up"]
        assert contract["referenceOrder"] == ["reference"]
        assert contract["outputs"] == {"deltas": {"dtype": "float32", "shape": [1, 1, 3]}}
        assert [value.name for value in graph.graph.output] == ["deltas"]
        assert not any("yolo" in tensor.name for tensor in graph.graph.initializer)
        assert contract["image"]["paddingValue"] == 114
        assert contract["image"]["scale"] == 1 / 255
        assert contract["contextOrder"] == ["normalized_position", "game", "map_name"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
        image = torch.rand(1, 3, imgsz, imgsz)
        references = torch.tensor([[[10., -20., 30.]]])
        context = torch.tensor([[0.25, 1., 1., 0.]])
        results = []
        for points, values in [(references, context), (references + 4, context),
                               (references, torch.tensor([[0.75, 1., 0., 1.]]))]:
            with torch.no_grad():
                expected = model(image, points, values)
            result = session.run(None, {"images": image.numpy(), "references": points.numpy(), "context": values.numpy()})
            assert len(result) == 1
            np.testing.assert_allclose(result[0], expected.numpy(), rtol=1e-4, atol=1e-5)
            results.append(result[0])
        for changed in results[1:]:
            assert not np.allclose(results[0], changed)
    finally:
        torch.set_num_threads(previous)
