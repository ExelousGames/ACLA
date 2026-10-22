"""Brand-independent DirectInput axes via the installed vJoy SDK (no Python package).

ABI: https://github.com/shauleiz/vJoy/blob/master/SDK/inc/vjoyinterface.h
X = steering, Y = throttle, Z = brake; buttons 1/2 = shift up/down.
This is a separate device, not physical wheel input.
"""
import ctypes
import os
from pathlib import Path

AXES = {"steering": 0x30, "throttle": 0x31, "brake": 0x32}
BUTTONS = {"shiftUp": 1, "shiftDown": 2}


def load_vjoy():
    architecture = "x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "x86"
    override = os.environ.get("ACLA_VJOY_DLL")
    candidates = [Path(override)] if override else [
        Path(root) / "vJoy" / architecture / "vJoyInterface.dll"
        for root in dict.fromkeys(filter(None, [os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles"),
                                                os.environ.get("ProgramFiles(x86)")]))
    ]
    dll_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if dll_path is None:
        raise RuntimeError(f"vJoyInterface.dll ({architecture}) was not found. For a custom install, set ACLA_VJOY_DLL to its full path.")
    dll = ctypes.CDLL(str(dll_path))  # vJoy exports __cdecl functions, including on 32-bit Windows.
    uint, boolean, long = ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32
    for name, result, arguments in [
        ("vJoyEnabled", boolean, []),
        ("GetVJDStatus", ctypes.c_int, [uint]),
        ("AcquireVJD", boolean, [uint]),
        ("RelinquishVJD", None, [uint]),
        ("GetVJDAxisExist", boolean, [uint, uint]),
        ("GetVJDAxisMin", boolean, [uint, uint, ctypes.POINTER(long)]),
        ("GetVJDAxisMax", boolean, [uint, uint, ctypes.POINTER(long)]),
        ("SetAxis", boolean, [long, uint, uint]),
        ("GetVJDButtonNumber", ctypes.c_int, [uint]),
        ("SetBtn", boolean, [boolean, uint, ctypes.c_ubyte]),
    ]:
        function = getattr(dll, name)
        function.restype = result
        function.argtypes = arguments
    return dll


class VJoyController:
    def __init__(self, device_id=1, sdk=None):
        if type(device_id) is not int or not 1 <= device_id <= 16:
            raise ValueError("vJoy device ID must be from 1 to 16.")
        self.sdk = sdk if sdk is not None else load_vjoy()
        self.device_id = device_id
        self.acquired = False
        self.ranges = {}
        if not self.sdk.vJoyEnabled():
            raise RuntimeError("The vJoy driver is disabled or unavailable. Enable it in Configure vJoy.")
        # VJD_STAT_FREE = 1; do not take a device owned by another feeder.
        if self.sdk.GetVJDStatus(device_id) != 1:
            raise RuntimeError(f"vJoy device {device_id} is busy or not configured. Select an enabled, unused device.")
        for name, axis in AXES.items():
            low, high = ctypes.c_int32(), ctypes.c_int32()
            if (not self.sdk.GetVJDAxisExist(device_id, axis)
                    or not self.sdk.GetVJDAxisMin(device_id, axis, ctypes.byref(low))
                    or not self.sdk.GetVJDAxisMax(device_id, axis, ctypes.byref(high)) or high.value <= low.value):
                raise RuntimeError(f"Enable X, Y and Z axes on vJoy device {device_id}; {name} is unavailable.")
            self.ranges[name] = (low.value, high.value)
        if self.sdk.GetVJDButtonNumber(device_id) < max(BUTTONS.values()):
            raise RuntimeError(f"Enable at least 2 buttons on vJoy device {device_id} for shift up and shift down.")
        if not self.sdk.AcquireVJD(device_id):
            raise RuntimeError(f"Could not acquire vJoy device {device_id}. Close other feeders or select another device.")
        self.acquired = True

    def _set_axis(self, name, value):
        low, high = self.ranges[name]
        normalized = (value + 1) / 2 if name == "steering" else value
        position = round(low + normalized * (high - low))
        if not self.sdk.SetAxis(position, self.device_id, AXES[name]):
            raise RuntimeError(f"vJoy device {self.device_id} stopped accepting {name} input. Reconnect to continue.")

    def _set_button(self, name, pressed):
        if not self.sdk.SetBtn(pressed, self.device_id, BUTTONS[name]):
            raise RuntimeError(f"vJoy device {self.device_id} stopped accepting {name} input. Reconnect to continue.")

    def apply(self, action):
        for name in AXES:
            self._set_axis(name, action[name])
        # Release the opposite paddle first when switching directions.
        for pressed in (False, True):
            for name in BUTTONS:
                if action.get(name, False) == pressed:
                    self._set_button(name, pressed)

    def reset(self):
        # ResetVJD centers every axis, which would leave both pedals half pressed.
        # Attempt all controls even if one fails, so pedals and paddles are still released.
        error = None
        for name in AXES:
            try:
                self._set_axis(name, 0)
            except RuntimeError as reason:
                error = reason
        for name in BUTTONS:
            try:
                self._set_button(name, False)
            except RuntimeError as reason:
                error = reason
        if error:
            raise error

    def close(self):
        if self.acquired:
            self.sdk.RelinquishVJD(self.device_id)
            self.acquired = False
