"""Compatibility wrapper for deterministic lap annotation."""

from app.local_annotation_agent.workflow.deterministic import calculate_lap_annotation


def calculate(**kwargs):
    return calculate_lap_annotation(**kwargs)


__all__ = ["calculate"]
