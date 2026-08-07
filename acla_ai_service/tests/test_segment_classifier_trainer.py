import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import app.ml.segment_classifier.trainer as trainer_module
from app.ml.segment_classifier.trainer import SegmentClassifierTrainer
from app.ml.segment_classifier.service import SegmentClassifierService


def test_rocm_training_enables_miopen_immediate_mode(monkeypatch, capsys):
    trainer = SegmentClassifierTrainer.__new__(SegmentClassifierTrainer)
    trainer.device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.backends.miopen, "immediate", False)

    trainer._configure_training_backend()

    assert torch.backends.miopen.immediate is True
    assert "Enabled MIOpen Immediate Mode" in capsys.readouterr().out


def test_non_rocm_training_does_not_change_miopen_mode(monkeypatch, capsys):
    trainer = SegmentClassifierTrainer.__new__(SegmentClassifierTrainer)
    monkeypatch.setattr(torch.backends.miopen, "immediate", False)

    trainer.device = torch.device("cuda")
    monkeypatch.setattr(torch.version, "hip", None)
    trainer._configure_training_backend()

    trainer.device = torch.device("cpu")
    monkeypatch.setattr(torch.version, "hip", "7.2")
    trainer._configure_training_backend()

    assert torch.backends.miopen.immediate is False
    assert capsys.readouterr().out == ""


def _preprocessor_trainer(targets):
    trainer = SegmentClassifierTrainer.__new__(SegmentClassifierTrainer)
    trainer.device = torch.device("cpu")
    trainer.classifier_service = SimpleNamespace(
        label_ids=["MSP", "MSP1"],
        behavior_label_ids=["MSP"],
    )
    sequence = SimpleNamespace(
        features=np.array([[40.0, 0.0], [41.0, 1.0]], dtype=np.float32),
        targets=np.array(targets, dtype=np.float32),
        loss_mask=np.ones((2, 2), dtype=np.float32),
    )
    trainer._iter_sequences = lambda cache_key: iter([sequence])
    trainer.scaler = None
    return trainer


@pytest.mark.asyncio
async def test_parent_only_training_warns_and_fits_preprocessors(caplog):
    trainer = _preprocessor_trainer([[1.0, 0.0], [0.0, 0.0]])

    with caplog.at_level("WARNING", logger=trainer_module.LOGGER.name):
        await trainer.fit_preprocessors("train")

    assert trainer.scaler is not None
    assert "training will continue with parent behavior labels only" in caplog.text
    assert "child-label predictions from this model will not be reliable" in caplog.text


@pytest.mark.asyncio
async def test_child_annotated_training_does_not_warn(caplog):
    trainer = _preprocessor_trainer([[1.0, 1.0], [0.0, 0.0]])

    with caplog.at_level("WARNING", logger=trainer_module.LOGGER.name):
        await trainer.fit_preprocessors("train")

    assert "No behavior sub-label annotations" not in caplog.text


@pytest.mark.asyncio
async def test_preprocessors_set_positive_weights_from_training_targets():
    trainer = _preprocessor_trainer([[1.0, 1.0], [0.0, 0.0]])
    sequence = next(trainer._iter_sequences("train"))
    sequence.targets = np.array(
        [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    sequence.features = np.zeros((4, 2), dtype=np.float32)
    sequence.loss_mask = np.ones((4, 2), dtype=np.float32)

    await trainer.fit_preprocessors("train")

    assert trainer.pos_weight.tolist() == pytest.approx([3.0, 1.0])


def test_masked_loss_applies_positive_class_weights():
    logits = torch.tensor([[[0.2, -0.4], [1.3, -2.0]]])
    targets = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    mask = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
    pos_weight = torch.tensor([2.0, 3.0])

    loss = SegmentClassifierTrainer._masked_loss(logits, targets, mask, pos_weight)
    expected = (F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    ) * mask).sum() / mask.sum()

    assert loss.item() == pytest.approx(expected.item())


@pytest.mark.asyncio
async def test_preprocessors_use_neutral_weight_for_label_without_positives():
    trainer = _preprocessor_trainer([[1.0, 0.0], [0.0, 0.0]])

    await trainer.fit_preprocessors("train")

    assert trainer.pos_weight.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.asyncio
async def test_preprocessors_cap_positive_weight_at_twenty():
    targets = [[1.0, 1.0], *([[0.0, 0.0]] * 24)]
    trainer = _preprocessor_trainer(targets)
    sequence = next(trainer._iter_sequences("train"))
    sequence.features = np.zeros((25, 2), dtype=np.float32)
    sequence.loss_mask = np.ones((25, 2), dtype=np.float32)

    await trainer.fit_preprocessors("train")

    assert trainer.pos_weight.tolist() == pytest.approx([20.0, 20.0])


def test_masked_class_accuracy_counts_positive_and_negative_predictions():
    logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]])
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])

    positive_counts, negative_counts = SegmentClassifierTrainer._masked_class_accuracy_counts(
        logits,
        targets,
        mask,
        torch.ones(2),
    )

    assert positive_counts == (1, 2)
    assert negative_counts == (1, 1)


def test_masked_class_accuracy_counts_use_corrected_weighted_logits():
    logits = torch.tensor([[[1.0, -0.5]]])
    targets = torch.tensor([[[0.0, 1.0]]])
    mask = torch.ones_like(targets)

    positive_counts, negative_counts = SegmentClassifierTrainer._masked_class_accuracy_counts(
        logits,
        targets,
        mask,
        torch.tensor([20.0, 0.25]),
    )

    assert positive_counts == (1, 1)
    assert negative_counts == (1, 1)


def test_masked_label_accuracy_counts_report_each_label_separately():
    logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]])
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])

    positive_counts, negative_counts = SegmentClassifierTrainer._masked_label_accuracy_counts(
        logits,
        targets,
        mask,
        torch.ones(2),
    )

    assert positive_counts == [(1, 1), (0, 1)]
    assert negative_counts == [(0, 0), (1, 1)]


def test_accuracy_percentage_is_unavailable_without_class_examples():
    assert SegmentClassifierTrainer._accuracy_percentage(0, 0) == "N/A"
    assert SegmentClassifierTrainer._accuracy_percentage(3, 4) == "75.00%"


def test_inference_service_has_no_training_entrypoint():
    assert not hasattr(SegmentClassifierService, "train_model")


@pytest.mark.asyncio
async def test_training_runs_all_epochs_and_restores_best_loss_state(monkeypatch, capsys):
    class CountingModel(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.output_dim = kwargs["output_dim"]

        def forward(self, features):
            return self.weight.expand(
                features.shape[0],
                features.shape[1],
                self.output_dim,
            )

    optimizers = []

    class CountingOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameter = next(iter(parameters))
            self.step_count = 0
            optimizers.append(self)

        def zero_grad(self):
            self.parameter.grad = None

        def step(self):
            with torch.no_grad():
                self.parameter.add_(1.0)
            self.step_count += 1

    class NoOpScheduler:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, loss):
            pass

    class ClassifierStub:
        device = torch.device("cpu")
        label_ids = ["MSP", "EA"]
        hidden_dim = 1
        dilations = (1,)
        dropout = 0.0
        model = None
        scaler = None
        artifacts_saved = False
        saved_label_weights = None

        @property
        def threshold(self):
            raise AssertionError("training accessed the inference threshold")

        def _save_artifacts(self):
            self.artifacts_saved = True
            self.saved_label_weights = dict(self.label_weights)

        def serialize_artifacts(self):
            return {}

    batch = (
        torch.zeros(1, 2, 1),
        torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    validation_losses = [1.0, 2.0, 3.0, 4.0, 5.0]
    classifier = ClassifierStub()
    trainer = SegmentClassifierTrainer.__new__(SegmentClassifierTrainer)
    trainer.classifier_service = classifier
    trainer.device = torch.device("cpu")
    trainer.scaler = SimpleNamespace(mean_=np.zeros(1))
    trainer.pos_weight = torch.tensor([2.0, 3.0])
    trainer.model = None
    trainer._configure_training_backend = lambda: None
    trainer.prepare_training_data = AsyncMock()
    trainer.fit_preprocessors = AsyncMock()
    trainer._dataset = lambda cache_key: cache_key

    def controlled_loss(logits, targets, mask, pos_weight):
        if trainer.model.training:
            return logits.mean() * 0 + 1.0
        return torch.tensor(validation_losses.pop(0))

    trainer._masked_loss = controlled_loss
    monkeypatch.setattr(trainer_module, "DataLoader", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(trainer_module, "TemporalDetectionModel", CountingModel)
    monkeypatch.setattr(torch.optim, "Adam", CountingOptimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", NoOpScheduler)

    backend_client = ModuleType("app.integrations.backend.client")
    backend_client.backend_service = SimpleNamespace(save_ai_model=AsyncMock())
    monkeypatch.setitem(sys.modules, "app.integrations.backend.client", backend_client)

    await trainer.train_model(epochs=5, annotation_cache_key="annotations")

    assert optimizers[0].step_count == 5
    assert validation_losses == []
    assert trainer.model.weight.item() == 1.0
    assert classifier.model is trainer.model
    assert classifier.scaler is trainer.scaler
    assert classifier.artifacts_saved is True
    assert classifier.saved_label_weights == {"MSP": 2.0, "EA": 3.0}
    report = capsys.readouterr().out
    assert "Train Loss:" in report
    assert "Val Loss:" in report
    assert "Val Samples:" in report
    assert "Val Accuracy: 50.00% (1/2 labeled predictions)" in report
    assert "Positive Accuracy: 100.00% (1/1)" in report
    assert "Negative Accuracy: 0.00% (0/1)" in report
    assert (
        "Label MSP: Positive Validation Accuracy: 100.00% (1/1), "
        "Negative Validation Accuracy: 0.00% (0/1)"
    ) in report
    assert (
        "Label EA: Positive Validation Accuracy: N/A (0/0), "
        "Negative Validation Accuracy: N/A (0/0)"
    ) in report
    assert (
        "Best validation result: epoch=1 loss=1.0000 accuracy=50.00% "
        "(1/2 labeled predictions) positive_accuracy=100.00% (1/1) "
        "negative_accuracy=0.00% (0/1)"
    ) in report
    assert "precision" not in report.lower()
    assert "recall" not in report.lower()
