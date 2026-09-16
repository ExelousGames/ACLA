"""Telemetry inputs allowed in automatic annotation."""

import pandas as pd


def annotation_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Copy telemetry without wheel slip channels or their derived metrics."""
    excluded = [
        column for column in df.columns
        if column.startswith(("Physics_slip_angle_", "Physics_slip_ratio_"))
        or column in {"driver_push_to_limit", "slip_balance"}
    ]
    return df.drop(columns=excluded)
