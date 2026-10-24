"""Read physical controllers through SDL (HID, DirectInput and XInput).

Use the joystick interface, not SDL's standardized game-controller mappings:
wheels, separate pedals and unrecognized controller layouts must work too.
"""
import ctypes as ct
from contextlib import redirect_stdout
import sys
from uuid import uuid4


def decode(value):
    return value.decode("utf-8", errors="replace") if value else ""


class PhysicalControllers:
    def __init__(self, api=None):
        if api is None:
            try:
                # Library diagnostics must not corrupt the JSON-lines protocol.
                with redirect_stdout(sys.stderr):
                    import sdl2 as api
            except (ImportError, RuntimeError) as error:
                raise RuntimeError("Controller input support is missing. Run npm run setup:python "
                                   "for development or reinstall the updated desktop app.") from error
        self.api = api
        self.handles = {}
        self.session_id = uuid4().hex
        self.closed = False
        # SDL's Windows RawInput backend delivers sticks/buttons through window
        # messages, while triggers may be polled through XInput. Its native event
        # pump needs the video subsystem even though we never create a window.
        self.subsystems = api.SDL_INIT_VIDEO | api.SDL_INIT_JOYSTICK
        api.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")
        if api.SDL_InitSubSystem(self.subsystems) != 0:
            raise RuntimeError("Cannot initialize controller input: " + decode(api.SDL_GetError()))
        # Read state directly; SDL still dispatches native messages when pumped.
        api.SDL_JoystickEventState(api.SDL_DISABLE)

    def read(self):
        # Run on the same thread as initialization (the bridge's main thread).
        self.api.SDL_PumpEvents()
        self.api.SDL_FlushEvents(self.api.SDL_FIRSTEVENT, self.api.SDL_LASTEVENT)
        # SDL can discover hardware asynchronously. Keep indices stable between
        # inspecting a device (including the vJoy filter) and opening its handle.
        self.api.SDL_LockJoysticks()
        try:
            return self._read()
        finally:
            self.api.SDL_UnlockJoysticks()

    def _read(self):
        api = self.api
        api.SDL_JoystickUpdate()
        count = api.SDL_NumJoysticks()
        if count < 0:
            raise RuntimeError("Cannot enumerate controllers: " + decode(api.SDL_GetError()))
        for instance, handle in list(self.handles.items()):
            if not api.SDL_JoystickGetAttached(handle):
                api.SDL_JoystickClose(handle)
                del self.handles[instance]
        for index in range(count):
            instance = api.SDL_JoystickGetDeviceInstanceID(index)
            if instance < 0 or instance in self.handles:
                continue
            name = decode(api.SDL_JoystickNameForIndex(index))
            vendor = api.SDL_JoystickGetDeviceVendor(index)
            product = api.SDL_JoystickGetDeviceProduct(index)
            # Exclude all vJoy outputs, including renamed devices, to avoid feedback.
            if "vjoy" in name.lower() or (vendor, product) == (0x1234, 0xBEAD):
                continue
            handle = api.SDL_JoystickOpen(index)
            if handle:
                self.handles[instance] = handle
        api.SDL_JoystickUpdate()  # Also refresh newly opened devices before reading.
        devices = []
        for instance, handle in list(self.handles.items()):
            if not api.SDL_JoystickGetAttached(handle):
                api.SDL_JoystickClose(handle)
                del self.handles[instance]
                continue
            guid = ct.create_string_buffer(33)
            api.SDL_JoystickGetGUIDString(api.SDL_JoystickGetGUID(handle), guid, len(guid))
            # Enumeration indices change when other devices unplug. Prefer the
            # hardware path/serial; a GUID alone aliases identical controllers.
            identity = (decode(api.SDL_JoystickPath(handle))
                        or decode(api.SDL_JoystickGetSerial(handle))
                        or f"instance:{self.session_id}:{instance}")
            axes = {}
            for axis in range(api.SDL_JoystickNumAxes(handle)):
                raw = api.SDL_JoystickGetAxis(handle, axis)
                axes[f"axis{axis}"] = (raw / (32768 if raw < 0 else 32767) + 1) / 2
            buttons = [bool(api.SDL_JoystickGetButton(handle, button))
                       for button in range(api.SDL_JoystickNumButtons(handle))]
            # Expose hat/D-pad directions as mappable buttons after native buttons.
            for hat in range(api.SDL_JoystickNumHats(handle)):
                value = api.SDL_JoystickGetHat(handle, hat)
                buttons.extend(bool(value & direction) for direction in (1, 2, 4, 8))
            # A disconnect during a sample must never return stale or neutral input.
            if api.SDL_JoystickGetAttached(handle):
                devices.append({
                    "id": f"sdl:{decode(guid.value)}:{identity}",
                    "name": decode(api.SDL_JoystickName(handle)) or "Controller",
                    "axes": axes, "buttons": buttons,
                })
        return devices

    def close(self):
        if self.closed:
            return
        self.closed = True
        for handle in self.handles.values():
            self.api.SDL_JoystickClose(handle)
        self.handles.clear()
        self.api.SDL_QuitSubSystem(self.subsystems)
