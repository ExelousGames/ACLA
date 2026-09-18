"""Compatibility shim for the moved training pipeline config."""

from training.pipelines.training.config import TrainingPipelineConfig


PipelineConfig = TrainingPipelineConfig

__all__ = ["PipelineConfig", "TrainingPipelineConfig"]
