"""Compatibility shim for the moved training pipeline config."""

from app.pipelines.training.config import TrainingPipelineConfig


PipelineConfig = TrainingPipelineConfig

__all__ = ["PipelineConfig", "TrainingPipelineConfig"]
