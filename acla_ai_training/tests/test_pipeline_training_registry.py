import sys
from types import SimpleNamespace

import pytest

from training.annotation_providers import registry as providers
from training.pipelines.manifest import node_kinds


def test_pipeline_registry_excludes_llm_training_component():
    training_kinds = {
        spec.kind for spec in node_kinds.list_by_category("training")
    }

    assert "llm_training" not in training_kinds
    with pytest.raises(KeyError, match="Unknown node kind: llm_training"):
        node_kinds.get("llm_training")


@pytest.mark.parametrize("enabled", [None, "local_vlm,claude_cli,openai"])
def test_annotation_providers_exclude_local_models(monkeypatch, enabled):
    monkeypatch.setattr(providers.settings, "annotation_enabled_providers", enabled)
    monkeypatch.setattr(providers.settings, "annotation_openai_compatible_base_url", None)
    providers.list_annotation_providers.cache_clear()
    try:
        assert {provider.id for provider in providers.list_annotation_providers()} == {
            "claude_cli", "openai",
        }
        with pytest.raises(providers.ProviderConfigurationError, match="Unknown annotation provider 'local_vlm'"):
            providers.get_annotation_provider("local_vlm")
    finally:
        providers.list_annotation_providers.cache_clear()


def test_training_service_initializes_without_local_llm(monkeypatch, tmp_path):
    for module in (
        "training.local_llm", "training.llama", "llama_cpp", "peft",
        "transformers", "huggingface_hub",
    ):
        monkeypatch.setitem(sys.modules, module, None)

    from training.pipelines.training import full_dataset
    from training.pipelines.training.config import TrainingPipelineConfig

    store = SimpleNamespace(store_dir=tmp_path / "telemetry")
    monkeypatch.setattr(full_dataset, "get_shared_telemetry_store", lambda: store)
    config = TrainingPipelineConfig()
    service = full_dataset.Full_dataset_TelemetryMLService(
        models_directory=str(tmp_path / "models"), pipeline_config=config,
    )

    assert service.telemetry_store is store
    assert service.pipeline_config is config
    assert not hasattr(service, "llm_orchestrator")
    assert not (service.models_directory / "llm_adapters").exists()
