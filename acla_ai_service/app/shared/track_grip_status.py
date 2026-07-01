"""ACC track grip status enum.

Mirrors the ``ACC_TRACK_GRIP_STATUS`` values exposed by Assetto Corsa
Competizione's shared memory. Values are ordered from lowest to highest
grip-affecting moisture (GREEN = cold/dirty dry, FLOODED = standing water).
"""

from __future__ import annotations

from enum import IntEnum


class TrackGripStatus(IntEnum):
    ACC_GREEN = 0
    ACC_FAST = 1
    ACC_OPTIMUM = 2
    ACC_GREASY = 3
    ACC_DAMP = 4
    ACC_WET = 5
    ACC_FLOODED = 6
