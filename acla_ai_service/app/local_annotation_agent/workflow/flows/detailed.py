"""Compatibility wrapper for deterministic detailed annotation."""

from app.local_annotation_agent.workflow.deterministic import calculate_detailed_annotation


def calculate(**kwargs):
    return calculate_detailed_annotation(**kwargs)


__all__ = ["calculate"]
