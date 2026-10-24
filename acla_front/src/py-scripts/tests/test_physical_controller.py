import ctypes as ct
import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from physical_controller import PhysicalControllers


class PhysicalControllerTests(unittest.TestCase):
    """Exercise the real bundled SDL using in-process test devices, without vJoy."""

    def setUp(self):
        self.reader = PhysicalControllers()
        self.api = self.reader.api
        self.test_handles = []
        self.instances = []

    def tearDown(self):
        for handle in self.test_handles:
            self.api.SDL_JoystickClose(handle)
        for instance in self.instances:
            self.detach(instance)
        self.reader.close()

    def attach(self, name=b'Test controller', axes=6, buttons=16, hats=0, vendor=1, product=2):
        descriptor = self.api.SDL_VirtualJoystickDesc(
            version=1, type=self.api.SDL_JOYSTICK_TYPE_UNKNOWN, name=name,
            naxes=axes, nbuttons=buttons, nhats=hats, vendor_id=vendor, product_id=product)
        self.api.SDL_LockJoysticks()
        try:
            index = self.api.SDL_JoystickAttachVirtualEx(ct.byref(descriptor))
            self.assertGreaterEqual(index, 0, self.api.SDL_GetError())
            handle = self.api.SDL_JoystickOpen(index)
            self.assertTrue(handle)
            instance = self.api.SDL_JoystickInstanceID(handle)
        finally:
            self.api.SDL_UnlockJoysticks()
        self.instances.append(instance)
        self.test_handles.append(handle)
        return handle, instance

    def detach(self, instance):
        self.api.SDL_LockJoysticks()
        try:
            for index in range(self.api.SDL_NumJoysticks()):
                if self.api.SDL_JoystickGetDeviceInstanceID(index) == instance:
                    self.assertEqual(self.api.SDL_JoystickDetachVirtual(index), 0, self.api.SDL_GetError())
                    break
        finally:
            self.api.SDL_UnlockJoysticks()

    def devices(self):
        # Do not depend on what physical hardware the test host has connected.
        return [device for device in self.reader.read() if device['name'].startswith('Test')]

    def test_reads_generic_axes_independent_triggers_extra_buttons_and_hats(self):
        handle, _ = self.attach(axes=10, buttons=64, hats=1)
        self.api.SDL_JoystickSetVirtualAxis(handle, 0, 0)
        self.api.SDL_JoystickSetVirtualAxis(handle, 4, -32768)
        self.api.SDL_JoystickSetVirtualAxis(handle, 5, 32767)
        self.api.SDL_JoystickSetVirtualAxis(handle, 9, -16384)
        self.api.SDL_JoystickSetVirtualButton(handle, 63, 1)
        self.api.SDL_JoystickSetVirtualHat(handle, 0, self.api.SDL_HAT_RIGHTUP)
        device, = self.devices()
        self.assertEqual(len(device['axes']), 10)
        self.assertEqual(device['axes']['axis0'], 0.5)
        self.assertEqual(device['axes']['axis4'], 0)
        self.assertEqual(device['axes']['axis5'], 1)
        self.assertEqual(device['axes']['axis9'], 0.25)
        self.assertFalse(device['buttons'][0])
        self.assertTrue(device['buttons'][63])
        self.assertEqual(device['buttons'][64:], [True, True, False, False])
        self.assertEqual(self.api.SDL_GetHint(b'SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS'), b'1')
        self.assertEqual(self.api.SDL_WasInit(self.api.SDL_INIT_VIDEO), self.api.SDL_INIT_VIDEO)
        self.assertFalse(self.api.SDL_GetKeyboardFocus())  # No visible/focused SDL window.

    def test_excludes_vjoy_by_name_and_hardware_id(self):
        self.attach(name=b'Test vJoy Device')
        self.attach(name=b'Test renamed output', vendor=0x1234, product=0xBEAD)
        self.attach(name=b'Test wheel')
        self.assertEqual([device['name'] for device in self.devices()], ['Test wheel'])

    def test_hotplug_and_identical_devices_do_not_reassign_another_devices_mapping(self):
        self.assertEqual(self.devices(), [])
        _, first = self.attach()
        _, second = self.attach()
        before = self.devices()
        self.assertEqual(len({device['id'] for device in before}), 2)
        self.detach(first)
        remaining, = self.devices()
        self.assertEqual(remaining['id'], before[1]['id'])
        self.detach(second)
        self.assertEqual(self.devices(), [])
        self.attach()
        reconnected, = self.devices()
        # With no hardware path or serial, require remapping instead of aliasing.
        self.assertNotIn(reconnected['id'], [device['id'] for device in before])

    def test_hardware_path_preserves_mapping_after_reconnect(self):
        with patch.object(self.api, 'SDL_JoystickPath', return_value=b'usb:test-device'):
            _, instance = self.attach()
            before, = self.devices()
            self.detach(instance)
            self.assertEqual(self.devices(), [])
            self.attach()
            after, = self.devices()
            self.assertEqual(before['id'], after['id'])

    def test_scans_beyond_the_old_sixteen_device_limit(self):
        for index in range(18):
            self.attach(name=f'Test device {index}'.encode(), axes=1, buttons=0)
        self.assertEqual(len(self.devices()), 18)

    def test_devices_without_persistent_identity_require_remapping_in_a_new_session(self):
        self.attach()
        before, = self.devices()
        other = PhysicalControllers()
        try:
            after, = [device for device in other.read() if device['name'].startswith('Test')]
            self.assertNotEqual(before['id'], after['id'])
        finally:
            other.close()

    def test_disconnection_during_read_does_not_return_a_stale_sample(self):
        handle, instance = self.attach()
        self.devices()
        get_axis = self.api.SDL_JoystickGetAxis

        def disconnect(joystick, axis):
            if self.api.SDL_JoystickInstanceID(joystick) == instance:
                self.detach(instance)
            return get_axis(joystick, axis)

        with patch.object(self.api, 'SDL_JoystickGetAxis', side_effect=disconnect):
            self.assertEqual(self.devices(), [])
        self.assertFalse(self.api.SDL_JoystickGetAttached(handle))

    @unittest.skipUnless(sys.platform == 'win32', 'Exercises the Windows native message queue')
    def test_windows_messages_deliver_stick_and_button_input(self):
        handle, _ = self.attach(axes=1, buttons=1)
        self.devices()
        callback_type = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint64, ct.c_int64)
        library = ct.CDLL(self.api.get_dll_file())
        hook = library.SDL_SetWindowsMessageHook
        hook.argtypes, hook.restype = [callback_type, ct.c_void_p], None
        message = 0x8001  # A test message to this thread only, not another app.

        @callback_type
        def receive(_userdata, _window, kind, _wparam, _lparam):
            if kind == message:
                self.api.SDL_JoystickSetVirtualAxis(handle, 0, 32767)
                self.api.SDL_JoystickSetVirtualButton(handle, 0, 1)

        user32 = ct.WinDLL('user32', use_last_error=True)
        post = user32.PostThreadMessageW
        post.argtypes, post.restype = [ct.c_uint32, ct.c_uint32, ct.c_size_t, ct.c_ssize_t], ct.c_int
        thread_id = ct.WinDLL('kernel32').GetCurrentThreadId()
        hook(receive, None)
        try:
            self.assertTrue(post(thread_id, message, 0, 0), ct.get_last_error())
            # State polling alone cannot deliver input queued by Windows.
            with patch.object(self.api, 'SDL_PumpEvents'):
                before, = self.devices()
            self.assertEqual(before['axes']['axis0'], 0.5)
            self.assertEqual(before['buttons'], [False])
            after, = self.devices()
            self.assertEqual(after['axes']['axis0'], 1)
            self.assertEqual(after['buttons'], [True])
        finally:
            hook(callback_type(), None)


class ControllerSetupTests(unittest.TestCase):
    def test_closes_handles_once(self):
        api = Mock(SDL_INIT_VIDEO=0x20, SDL_INIT_JOYSTICK=0x200)
        api.SDL_InitSubSystem.return_value = 0
        reader = PhysicalControllers(api)
        handle = object()
        reader.handles[1] = handle
        reader.close()
        reader.close()
        api.SDL_JoystickClose.assert_called_once_with(handle)
        api.SDL_QuitSubSystem.assert_called_once_with(api.SDL_INIT_VIDEO | api.SDL_INIT_JOYSTICK)

    def test_missing_runtime_reports_setup_action(self):
        with patch.dict(sys.modules, {'sdl2': None}):
            with self.assertRaisesRegex(RuntimeError, 'npm run setup:python'):
                PhysicalControllers()

    def test_initialization_failure_is_reported(self):
        api = Mock(SDL_INIT_VIDEO=0x20, SDL_INIT_JOYSTICK=0x200)
        api.SDL_InitSubSystem.return_value = -1
        api.SDL_GetError.return_value = b'Driver failed'
        with self.assertRaisesRegex(RuntimeError, 'Driver failed'):
            PhysicalControllers(api)


if __name__ == '__main__':
    unittest.main()
