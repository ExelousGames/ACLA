"""Spatial recurrence, checkpoint restoration, and the unchanged position ONNX interface."""

import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch import nn

from training.image_segmentation import position_model_v2, position_publication, position_training
from training.image_segmentation.__main__ import main
from training.image_segmentation.position_export import export_position_checkpoint
from training.image_segmentation.position_model import PositionModel
from training.image_segmentation.position_model_v2 import PositionModelV2


CATEGORIES = {"game": ["acc"], "map_name": ["Monza"]}


@pytest.fixture(autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        yield
    torch.set_num_threads(previous)


def model_inputs(batch, length, imgsz):
    values = (
        torch.rand(batch, length, 3, imgsz, imgsz),
        torch.rand(batch, length, 1, 3),
        torch.cat((torch.rand(batch, length, 1), torch.ones(batch, length, 2)), dim=-1),
        torch.rand(batch, length, 3),
    )
    return values if length > 1 else tuple(value[:, 0] for value in values)


def test_training_defaults_to_pretrained_resnet_with_spatial_features_and_internal_normalization(monkeypatch):
    backbone = position_model_v2.resnet18(weights=None)
    weights_seen = []

    def encoder(*, weights):
        weights_seen.append(weights)
        return backbone

    monkeypatch.setattr(position_model_v2, "resnet18", encoder)
    assert position_training.PositionModel is PositionModel
    model = PositionModelV2(96, categories=CATEGORIES, sequence_length=4)
    assert weights_seen == [position_model_v2.ResNet18_Weights.IMAGENET1K_V1]
    assert model.image_encoder[0] is backbone.conv1
    images = torch.rand(2, 3, 96, 96)
    received = []
    hook = model.image_encoder[0].register_forward_pre_hook(lambda module, args: received.append(args[0]))
    features = model.visual_features(images)
    hook.remove()
    assert features.shape == (2, 512, 3, 3)
    torch.testing.assert_close(received[0], (images - model.image_mean) / model.image_std)


@pytest.mark.parametrize("length", [1, 4])
def test_training_updates_projection_memory_and_head_with_frozen_resnet(length):
    model = PositionModelV2(32, categories=CATEGORIES, sequence_length=length, pretrained=False).train()
    batchnorm = [layer for layer in model.image_encoder.modules() if isinstance(layer, nn.BatchNorm2d)]
    encoder_state = {name: value.clone() for name, value in model.image_encoder.state_dict().items()}
    trained_layers = (model.position_encoder, model.temporal_encoder, model.head)
    before = [{name: value.clone() for name, value in layer.state_dict().items()} for layer in trained_layers]
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4)
    for mode in (False, True, False, True):
        model.train(mode)
        assert all(not layer.training for layer in model.image_encoder.modules())
        assert all(layer.training == mode for layer in trained_layers)
    # Batch one and a 1x1 final feature map must train, including a partial last batch.
    inputs = model_inputs(1, length, 32)
    inputs[0].requires_grad_()
    inputs[3].requires_grad_()
    expected = model(*inputs)
    assert expected.shape == (1, 1, 3)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(model(*inputs), expected)
    if length > 1:
        for field in range(4):
            changed = [value.detach().clone() for value in inputs]
            changed[field][:, :-1] = 0
            assert not torch.allclose(model(*changed), expected, rtol=1e-6, atol=1e-7)
        reordered = [torch.cat((value[:, :3].flip(1), value[:, 3:]), dim=1) for value in inputs]
        assert not torch.allclose(model(*reordered), expected, rtol=1e-6, atol=1e-7)
    expected.square().mean().backward()
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.image_encoder.parameters())
    for parameter in (model.position_encoder[0].weight, model.temporal_encoder.gates.weight, model.head[-1].weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert inputs[0].grad.abs().sum() > 0
    assert inputs[3].grad.abs().sum() > 0
    if length > 1:
        assert inputs[0].grad[:, 0].abs().sum() > 0
        assert inputs[3].grad[:, 0].abs().sum() > 0
    optimizer.step()
    for name, value in model.image_encoder.state_dict().items():
        assert torch.equal(value, encoder_state[name]), name
    for layer, state in zip(trained_layers, before):
        assert any(not torch.equal(value, state[name]) for name, value in layer.state_dict().items())
    for layer in batchnorm:
        assert not layer.training and not layer.weight.requires_grad and not layer.bias.requires_grad


@pytest.mark.parametrize("length,imgsz", [(1, 32), (4, 96), (2, 128)])
def test_checkpoint_and_onnx_preserve_inputs_outputs_and_window_isolation(tmp_path, monkeypatch, length, imgsz):
    encoder = position_model_v2.resnet18

    def offline_encoder(*, weights):
        assert weights is None, "Loading a checkpoint must never download pretrained weights"
        return encoder(weights=None)

    monkeypatch.setattr(position_model_v2, "resnet18", offline_encoder)
    model = PositionModelV2(imgsz, categories=CATEGORIES, sequence_length=length, pretrained=False).eval()
    model.position_mean.copy_(torch.tensor([4., -2., 1.]))
    model.position_std.copy_(torch.tensor([2., 3., 4.]))
    checkpoint = tmp_path / "best.pt"
    model.save(checkpoint, metrics={"val_mae": 0.25})
    restored = PositionModel.load(checkpoint)
    assert isinstance(restored, PositionModelV2)
    inputs = model_inputs(2, length, imgsz)
    with torch.no_grad():
        expected = model(*inputs)
        torch.testing.assert_close(restored(*inputs), expected)
        # The first sample cannot affect the second sample's recurrent state.
        torch.testing.assert_close(restored(*(value[1:] for value in inputs)), expected[1:])
    artifact = export_position_checkpoint(checkpoint)
    metadata = json.loads(artifact.with_suffix(".onnx.json").read_text())["metadata"]
    assert metadata["modelName"] == "position_model_v2"
    assert metadata["format"] == "acla-position-resnet18-convlstm-v1"
    assert metadata["metrics"] == {"val_mae": 0.25}
    contract = metadata["inference"]
    assert contract["schemaVersion"] == (7 if length > 1 else 6)
    assert contract["inputs"] == {
        name: {"dtype": "float32", "shape": [1, *value.shape[1:]]}
        for name, value in zip(("images", "references", "context", "orientation"), inputs)
    }
    assert contract["outputs"] == {"deltas": {"dtype": "float32", "shape": [1, 1, 3]}}
    assert contract["image"]["scale"] == 1 / 255
    if length > 1:
        assert contract["temporal"]["type"] == "convlstm"
        assert contract["temporal"]["sequenceLength"] == length
        assert contract["temporal"]["state"] == "reset-per-window"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
    for field in range(4):
        changed = [value[:1].clone() for value in inputs]
        if length > 1:
            changed[field][:, :-1] = 0
        else:
            changed[field].zero_()
        actual = session.run(None, dict(zip(contract["inputs"], [value.numpy() for value in changed])))
        with torch.no_grad():
            np.testing.assert_allclose(actual[0], model(*changed).numpy(), rtol=1e-4, atol=1e-5)
    # Both training publication and --upload-model use the artifact's model name.
    upload = AsyncMock(return_value={"_id": "v2-upload"})
    monkeypatch.setattr(position_publication, "upload_ultralytics_model", upload)
    asyncio.run(position_publication.upload_position_checkpoint(artifact, backend_service=object()))
    assert upload.call_args.args[3]["name"] == "position_model_v2"
    original_upload = position_publication.upload_position_checkpoint

    async def publish(path, **kwargs):
        return await original_upload(path, backend_service=object(), **kwargs)

    monkeypatch.setattr(position_publication, "upload_position_checkpoint", publish)
    assert main(["train-position", "--upload-model", str(artifact)]) == 0
    assert upload.call_args.args[3]["name"] == "position_model_v2"
    assert main(["train-position-v2", "--upload-model", str(artifact)]) == 0
    assert upload.call_args.args[3]["name"] == "position_model_v2"
    assert main(["upload-position", str(artifact)]) == 0
    assert upload.call_args.args[3]["name"] == "position_model_v2"


@pytest.mark.parametrize("field", range(4))
def test_temporal_model_rejects_misaligned_history(field):
    model = PositionModelV2(32, categories=CATEGORIES, sequence_length=4, pretrained=False)
    inputs = list(model_inputs(1, 4, 32))
    inputs[field] = inputs[field][:, :-1]
    with pytest.raises(ValueError, match="Temporal"):
        model(*inputs)
