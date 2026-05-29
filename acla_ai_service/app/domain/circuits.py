"""Circuit id → display name. Owned by the domain layer; the canonical
registry of which main-label ids are circuits.

Pure data, mirroring ``app.domain.circuit_sections``: a single mapping,
no behaviour. Section ranges live in ``circuit_sections``; section names
live in ``app.domain.labels.LABEL_MAPPING``.
"""

from __future__ import annotations

from typing import Dict

CIRCUIT_NAMES: Dict[str, str] = {
    "brands_hatch": "Brands Hatch",
    "silverstone": "Silverstone",
}
