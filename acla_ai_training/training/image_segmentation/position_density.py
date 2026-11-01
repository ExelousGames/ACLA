"""Select training endpoints by driver pose while retaining sparse coverage."""

from __future__ import annotations

import random

import numpy as np


def select_position_endpoints(records, *, position_radius=None, orientation_degrees=5.0, seed=42):
    """Return original-order indices, suppressing neighbors of retained poses only.

    A neighbor has the same game/map, Euclidean driver XYZ distance <= radius,
    and wrapped heading, pitch, roll differences each <= orientation_degrees.
    Randomized traversal avoids always favoring the first session. Only retained
    endpoints query the spatial index; no all-pairs distance matrix or neighbor
    graph is built. Callers must build temporal history BEFORE applying indices.
    """
    if not np.isfinite(orientation_degrees) or not 0 < orientation_degrees <= 180:
        raise ValueError("Density orientation degrees must be finite and in (0, 180].")
    if position_radius is None:
        return list(range(len(records)))
    if not np.isfinite(position_radius) or position_radius <= 0:
        raise ValueError("Density position radius must be finite and positive.")

    from scipy.spatial import cKDTree

    groups = {}
    for index, row in enumerate(records):
        groups.setdefault((row["game"], row["map_name"]), []).append(index)
    selected = []
    rng = random.Random(seed)
    angle_limit = np.deg2rad(orientation_degrees)
    for key in sorted(groups):
        indices = groups[key]
        vectors = {}
        for field in ("driver_position", "orientation"):
            values = []
            for index in indices:
                vector = np.asarray(records[index].get(field))
                if vector.shape != (3,) or vector.dtype.kind not in "fi" or not np.isfinite(vector).all():
                    raise ValueError(
                        f"Density sampling requires finite {field} XYZ/angle triples: "
                        f"{records[index]['image']}. Regenerate with prepare_position_dataset.py."
                    )
                values.append(vector)
            vectors[field] = np.asarray(values, dtype=np.float64)
        positions = vectors["driver_position"]
        angles = vectors["orientation"] % (2 * np.pi)
        tree = cKDTree(positions)
        suppressed = np.zeros(len(indices), dtype=bool)
        order = list(range(len(indices)))
        rng.shuffle(order)
        for current in order:
            if suppressed[current]:
                continue
            selected.append(indices[current])
            neighbors = np.asarray(tree.query_ball_point(positions[current], position_radius), dtype=np.intp)
            neighbors = neighbors[~suppressed[neighbors]]
            differences = np.abs((angles[neighbors] - angles[current] + np.pi) % (2 * np.pi) - np.pi)
            suppressed[neighbors[np.all(differences <= angle_limit, axis=1)]] = True
    return sorted(selected)
