"""Runtime-only access to the backend-owned top-lap reference model."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.top_laps.model import NoTopLapReferenceError, TopLapStore
from app.top_laps.shared import calculate_reference_features, deserialize_top_lap_store


class TopLapReferenceModelError(ValueError):
    """The runtime top-lap reference cannot satisfy an analysis request."""


class RuntimeTopLapReferenceModel:
    """Keep backend references in memory and enrich runtime telemetry."""

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.top_lap_store = TopLapStore(logger=self.logger)

    def reset(self) -> None:
        """Discard the in-memory reference and clear runtime readiness."""

        self.top_lap_store = TopLapStore(logger=self.logger)

    def install_backend_payload(self, payload: Dict[str, Any]) -> None:
        """Validate a backend payload before replacing the in-memory reference."""

        self.top_lap_store = deserialize_top_lap_store(
            payload,
            logger=self.logger,
        )
        self.logger.info(
            "Installed %d runtime top-lap reference entries",
            len(self.top_lap_store.entries),
        )

    def is_ready(self) -> bool:
        return bool(self.top_lap_store.entries)

    def enrich(
        self,
        records: List[Dict[str, Any]],
        track: Optional[str] = None,
        car: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return copied telemetry rows merged with all expert classifier fields."""

        if not self.is_ready():
            raise TopLapReferenceModelError(
                "Top-lap reference model is unavailable"
            )
        if not records:
            return []

        copied_rows = [dict(record) for record in records]
        for row in copied_rows:
            if self._is_missing(row.get("Static_track")) and not self._is_missing(
                track
            ):
                row["Static_track"] = track
            if self._is_missing(
                row.get("Static_car_model")
            ) and not self._is_missing(car):
                row["Static_car_model"] = car

        try:
            reference_rows = calculate_reference_features(
                self.top_lap_store,
                copied_rows,
            )
        except NoTopLapReferenceError as exc:
            raise TopLapReferenceModelError(
                f"No top-lap reference for track {exc.track!r} "
                f"and car {exc.car!r}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise TopLapReferenceModelError(str(exc)) from exc

        for row, reference_features in zip(copied_rows, reference_rows):
            row.update(reference_features)
        return copied_rows

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())


top_lap_reference_model = RuntimeTopLapReferenceModel()


__all__ = [
    "RuntimeTopLapReferenceModel",
    "TopLapReferenceModelError",
    "top_lap_reference_model",
]
