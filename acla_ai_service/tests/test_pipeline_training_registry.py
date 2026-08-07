import pytest

from app.local_llm.local_llm import LocalLLMConfig, LocalTelemetryLLM
from app.pipelines.manifest import node_kinds


def test_pipeline_registry_excludes_llm_training_component():
    training_kinds = {
        spec.kind for spec in node_kinds.list_by_category("training")
    }

    assert "llm_training" not in training_kinds
    with pytest.raises(KeyError, match="Unknown node kind: llm_training"):
        node_kinds.get("llm_training")


def test_local_llm_api_is_inference_only():
    config = LocalLLMConfig()

    assert not hasattr(config, "training")
    assert not hasattr(config, "lora")
    assert not hasattr(LocalTelemetryLLM, "train")
