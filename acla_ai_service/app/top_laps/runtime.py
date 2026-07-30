"""Runtime-only access to the backend-owned top-lap reference model."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.top_laps.model import NoTopLapReferenceError, TopLapStore
from app.top_laps.service import TopLapReferenceModelService
from app.top_laps.shared import calculate_reference_features


class TopLapReferenceModelError(ValueError):
    """The runtime top-lap reference cannot satisfy an analysis request."""


class RuntimeTopLapReferenceModel(TopLapReferenceModelService):
    """Install backend payloads and enrich runtime telemetry without training."""

    def __init__(
        self,
        artifact_path: Optional[Path] = None,
        *,
        logger: Optional[logging.Logger] = None,
    ):
        service_root = Path(__file__).resolve().parents[2]
        self.artifact_path = Path(
            artifact_path
            or (service_root / "top_lap_models" / "top_lap_store.json")
        )
        runtime_logger = logger or logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        super().__init__(logger=runtime_logger)

    def reset(self) -> None:
        """Clear runtime readiness without reading or deleting a local artifact."""

        self.top_lap_store = TopLapStore(logger=self.logger)

    def install_backend_payload(self, payload: Dict[str, Any]) -> None:
        """Validate, atomically persist, and activate a backend model payload."""

        candidate_service = TopLapReferenceModelService(
            logger=self.logger,
        )
        candidate_service.load_reference_model(payload)
        serialized_payload = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )

        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.artifact_path.parent,
                prefix=f".{self.artifact_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(serialized_payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.artifact_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        self.top_lap_store = candidate_service.top_lap_store
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
