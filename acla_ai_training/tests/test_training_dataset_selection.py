import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(params=[
    "train_segment_classifier",
    "train_segment_cropper",
    "train_transformer_guidance",
    "run_all_trainings",
])
def entrypoint(request, monkeypatch):
    module = importlib.import_module(
        f"training.pipelines.training.entrypoints.{request.param}"
    )
    training_calls = []
    for name in ("segment_classifier_trainer", "segment_cropper_trainer"):
        if hasattr(module, name):
            train = AsyncMock()
            monkeypatch.setattr(module, name, SimpleNamespace(train_model=train))
            training_calls.append(train)
    if hasattr(module, "run_transformer_guidance_training"):
        train = AsyncMock(return_value={"success": True})
        monkeypatch.setattr(module, "run_transformer_guidance_training", train)
        monkeypatch.setattr(module, "Full_dataset_TelemetryMLService", MagicMock())
        training_calls.append(train)
    return module, training_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [[], ["--annotation-key", ""], ["--annotation-key", "  "]])
async def test_training_command_rejects_missing_dataset(
    entrypoint, monkeypatch, capsys, arguments,
):
    module, training_calls = entrypoint
    monkeypatch.setattr(sys, "argv", [module.__file__, *arguments])

    with pytest.raises(SystemExit) as error:
        await module.main()

    assert error.value.code == 2
    assert "--annotation-key" in capsys.readouterr().err
    for train in training_calls:
        train.assert_not_called()


@pytest.mark.asyncio
async def test_training_command_passes_explicit_dataset(entrypoint, monkeypatch):
    module, training_calls = entrypoint
    monkeypatch.setattr(
        sys, "argv", [module.__file__, "--annotation-key", "selected_annotations"],
    )

    assert await module.main() == 0

    for train in training_calls:
        train.assert_awaited_once()
        call = train.await_args
        dataset = call.args[0] if call.args else call.kwargs["annotation_cache_key"]
        assert dataset == "selected_annotations"


@pytest.fixture(params=[
    ("segment_classifier", "SegmentClassifierTrainer"),
    ("segment_cropper", "SegmentCropperTrainer"),
])
def trainer(request):
    name, class_name = request.param
    module = importlib.import_module(f"training.ml.{name}.trainer")
    trainer_class = getattr(module, class_name)
    return trainer_class.__new__(trainer_class)


@pytest.mark.asyncio
async def test_trainer_requires_dataset_argument(trainer):
    with pytest.raises(TypeError, match="annotation_cache_key"):
        await trainer.train_model()


@pytest.mark.asyncio
@pytest.mark.parametrize("dataset", [None, "", "  "])
async def test_trainer_rejects_empty_dataset_before_training(trainer, dataset):
    with pytest.raises(ValueError, match="annotation dataset"):
        await trainer.train_model(annotation_cache_key=dataset)


def test_pipeline_creation_does_not_require_a_default_dataset(monkeypatch, tmp_path):
    from training.pipelines.manifest import registry

    monkeypatch.setattr(registry, "_STORAGE_ROOT", tmp_path)
    pipeline = registry.create_pipeline(name="Explicit inputs")

    assert pipeline.annotations == []
    assert pipeline.trainings == []
    assert registry.load(pipeline.id).to_dict() == pipeline.to_dict()
