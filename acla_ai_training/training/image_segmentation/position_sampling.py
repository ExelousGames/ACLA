"""Thin nearby position targets without constructing an all-pairs distance matrix."""

import random

import numpy as np
from scipy.spatial import KDTree


def select_position_samples(records, *, min_offset_distance=None, min_orientation_distance_deg=None,
                            min_normalized_position_distance=None, seed=42):
    """Return retained indices in input order, comparing only within each game/map.

    A seeded greedy pass suppresses neighbors of each retained sample. Neighbors
    must satisfy every enabled threshold: Euclidean local XYZ distance, the
    largest wrapped heading/pitch/roll difference, and circular lap distance.
    Thresholds are inclusive. The tree bounds candidates using XYZ and sine/cosine
    angles and lap positions; exact checks follow each query. Only one
    neighborhood is materialized at a time.
    """
    for name, value in (("min_offset_distance", min_offset_distance),
                        ("min_orientation_distance_deg", min_orientation_distance_deg),
                        ("min_normalized_position_distance", min_normalized_position_distance)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    if min_orientation_distance_deg is not None and min_orientation_distance_deg > 180:
        raise ValueError("min_orientation_distance_deg must be at most 180 degrees.")
    if min_normalized_position_distance is not None and min_normalized_position_distance > 0.5:
        raise ValueError("min_normalized_position_distance must be at most 0.5 laps.")
    if min_offset_distance is None and min_orientation_distance_deg is None and min_normalized_position_distance is None:
        return list(range(len(records)))

    groups = {}
    for index, row in enumerate(records):
        groups.setdefault((row["game"], row["map_name"]), []).append(index)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda index: str(records[index]["image"]))
        features = []
        if min_offset_distance is not None:
            offsets = np.asarray([records[index]["target"][0] for index in indices], dtype=np.float64)
            features.append(offsets / min_offset_distance)
        if min_orientation_distance_deg is not None:
            angles = np.asarray([records[index]["orientation"] for index in indices], dtype=np.float64)
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            angle_limit = np.deg2rad(min_orientation_distance_deg)
            # Each sine/cosine coordinate differs by no more than this chord.
            chord = 2 * np.sin(angle_limit / 2)
            features.extend((np.sin(angles) / chord, np.cos(angles) / chord))
        if min_normalized_position_distance is not None:
            positions = np.asarray([records[index]["normalized_position"] for index in indices], dtype=np.float64)
            lap_angles = 2 * np.pi * positions[:, None]
            lap_chord = 2 * np.sin(np.pi * min_normalized_position_distance)
            features.extend((np.sin(lap_angles) / lap_chord, np.cos(lap_angles) / lap_chord))
        points = np.concatenate(features, axis=1)
        tree = KDTree(points)
        available = np.ones(len(indices), dtype=bool)
        order = list(range(len(indices)))
        rng.shuffle(order)
        for index in order:
            if not available[index]:
                continue
            selected.append(indices[index])
            # A slightly expanded bound avoids losing boundary neighbors to
            # rounding in the scaled coordinates. Exact distances decide below.
            neighbors = np.asarray(tree.query_ball_point(points[index], 1 + 1e-12, p=np.inf), dtype=np.intp)
            neighbors = neighbors[available[neighbors]]
            close = np.ones(len(neighbors), dtype=bool)
            if min_offset_distance is not None:
                close &= np.linalg.norm(offsets[neighbors] - offsets[index], axis=1) <= min_offset_distance
            if min_orientation_distance_deg is not None:
                difference = np.abs((angles[neighbors] - angles[index] + np.pi) % (2 * np.pi) - np.pi)
                close &= np.max(difference, axis=1) <= angle_limit
            if min_normalized_position_distance is not None:
                difference = np.abs(positions[neighbors] - positions[index])
                close &= np.minimum(difference, 1 - difference) <= min_normalized_position_distance
            available[neighbors[close]] = False
    return sorted(selected)
