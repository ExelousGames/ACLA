"""Shared feature preparation for cropper training and inference."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def prepare_feature_frame(
    telemetry_data: Sequence[dict] | pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    source = (
        telemetry_data.reset_index(drop=True)
        if isinstance(telemetry_data, pd.DataFrame)
        else pd.DataFrame(telemetry_data)
    )
    frame = source.reindex(columns=list(feature_names), fill_value=0)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0)
    differences = frame.diff().fillna(0).add_suffix("_diff")
    return pd.concat([frame, differences], axis=1)


__all__ = ["prepare_feature_frame"]
