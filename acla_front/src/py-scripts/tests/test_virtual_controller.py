import queue
import os
import sys
import unittest
from unittest.mock import Mock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from virtual_controller import XboxController, create_controller, serve, valid_action
from virtual_wheel import AXES, VJoyController


class ControllerTests(unittest.TestCase):
    def test_applies_controls_and_releases_on_eof(self):
        gamepad = Mock()
        commands = queue.Queue()
        commands.put({"id": 1, "action": {"throttle": 0.4, "brake": 0.1, "steering": -0.5}})
        commands.put(None)
        messages = []
        serve(XboxController(gamepad), commands, messages.append)
        gamepad.right_trigger_float.assert_called_once_with(value_float=0.4)
        gamepad.left_trigger_float.assert_called_once_with(value_float=0.1)
        gamepad.left_joystick_float.assert_called_once_with(x_value_float=-0.5, y_value_float=0.0)
        self.assertEqual(gamepad.reset.call_count, 2)
        self.assertEqual(messages[-1], {"type": "applied", "id": 1})

    def test_watchdog_resets_without_new_commands(self):
        gamepad = Mock()

        class StalledQueue:
            now = 0

            def get(self, timeout):
                self.now += 0.1
                if self.now == 0.1:
                    return {"id": 1, "action": {"throttle": 0.8, "brake": 0, "steering": 0.2}}
                raise queue.Empty

        commands = StalledQueue()
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            serve(XboxController(gamepad), commands, lambda _: None, lambda: commands.now)
        self.assertEqual(gamepad.reset.call_count, 2)
        self.assertEqual(gamepad.update.call_count, 3)

    def test_invalid_commands_reset_device(self):
        for value in [float("nan"), float("inf"), -0.1, 2, True, "0.5"]:
            action = {"throttle": value, "brake": 0, "steering": 0}
            self.assertFalse(valid_action(action))
            commands = queue.Queue()
            commands.put({"action": action})
            gamepad = Mock()
            with self.assertRaises(ValueError):
                serve(XboxController(gamepad), commands, lambda _: None)
            self.assertEqual(gamepad.reset.call_count, 2)


def wheel_sdk(low=0, high=32768):
    sdk = Mock()
    sdk.GetVJDStatus.return_value = 1

    def limit(value):
        def write(_device, _axis, pointer):
            pointer._obj.value = value
            return True
        return write

    sdk.GetVJDAxisMin.side_effect = limit(low)
    sdk.GetVJDAxisMax.side_effect = limit(high)
    return sdk


class WheelControllerTests(unittest.TestCase):
    def test_wheel_backend_does_not_create_an_xbox_device(self):
        with patch("virtual_wheel.VJoyController") as wheel, patch.dict(sys.modules, {"vgamepad": None}):
            self.assertIs(create_controller("vjoy", 4), wheel.return_value)
            wheel.assert_called_once_with(4)

    def test_independent_pedals_full_steering_range_and_neutral(self):
        sdk = wheel_sdk()
        controller = VJoyController(3, sdk)
        controller.apply({"steering": -1, "throttle": 1, "brake": 0.25})
        controller.apply({"steering": 1, "throttle": 0, "brake": 1})
        controller.reset()
        self.assertEqual(sdk.SetAxis.call_args_list, [
            call(0, 3, AXES["steering"]), call(32768, 3, AXES["throttle"]), call(8192, 3, AXES["brake"]),
            call(32768, 3, AXES["steering"]), call(0, 3, AXES["throttle"]), call(32768, 3, AXES["brake"]),
            call(16384, 3, AXES["steering"]), call(0, 3, AXES["throttle"]), call(0, 3, AXES["brake"]),
        ])

    def test_uses_driver_reported_axis_range(self):
        sdk = wheel_sdk(100, 1100)
        controller = VJoyController(1, sdk)
        controller.apply({"steering": 0.2, "throttle": 0.5, "brake": 0})
        self.assertEqual(sdk.SetAxis.call_args_list, [call(700, 1, 0x30), call(600, 1, 0x31), call(100, 1, 0x32)])

    def test_resets_before_ready_and_releases_on_eof(self):
        sdk = wheel_sdk()
        commands = queue.Queue()
        commands.put({"id": 12, "action": {"steering": 0.5, "throttle": 1, "brake": 0}})
        commands.put(None)
        messages = []

        def receive(message):
            if message["type"] == "ready":
                self.assertEqual(sdk.SetAxis.call_count, 3)
            messages.append(message)

        serve(VJoyController(1, sdk), commands, receive)
        self.assertEqual(messages, [{"type": "ready"}, {"type": "applied", "id": 12}])
        self.assertEqual(sdk.mock_calls[-4:], [call.SetAxis(16384, 1, 0x30), call.SetAxis(0, 1, 0x31),
                                             call.SetAxis(0, 1, 0x32), call.RelinquishVJD(1)])

    def test_watchdog_neutralizes_and_relinquishes_wheel(self):
        class StalledQueue:
            now = 0

            def get(self, timeout):
                self.now += 0.1
                if self.now == 0.1:
                    return {"action": {"steering": 1, "throttle": 1, "brake": 0}}
                raise queue.Empty

        sdk, commands = wheel_sdk(), StalledQueue()
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            serve(VJoyController(1, sdk), commands, lambda _: None, lambda: commands.now)
        self.assertEqual(sdk.mock_calls[-4:], [call.SetAxis(16384, 1, 0x30), call.SetAxis(0, 1, 0x31),
                                             call.SetAxis(0, 1, 0x32), call.RelinquishVJD(1)])

    def test_driver_and_axis_failures_do_not_claim_device(self):
        for method, value, message in [("vJoyEnabled", False, "disabled"), ("GetVJDStatus", 2, "busy"),
                                       ("GetVJDAxisExist", False, "Enable X"), ("AcquireVJD", False, "acquire")]:
            sdk = wheel_sdk()
            getattr(sdk, method).return_value = value
            with self.assertRaisesRegex(RuntimeError, message):
                VJoyController(1, sdk)
            sdk.SetAxis.assert_not_called()
            sdk.RelinquishVJD.assert_not_called()

    def test_failed_axis_write_is_not_acknowledged_and_other_axes_release(self):
        sdk = wheel_sdk()
        # The initial reset succeeds. Steering fails when the action is applied and on reset.
        sdk.SetAxis.side_effect = [True, True, True, False, False, True, True]
        commands = queue.Queue()
        commands.put({"id": 2, "action": {"steering": 1, "throttle": 1, "brake": 0}})
        messages = []
        with self.assertRaisesRegex(RuntimeError, "stopped accepting"):
            serve(VJoyController(1, sdk), commands, messages.append)
        self.assertEqual(messages, [{"type": "ready"}])
        self.assertEqual(sdk.mock_calls[-3:], [call.SetAxis(0, 1, 0x31), call.SetAxis(0, 1, 0x32), call.RelinquishVJD(1)])


if __name__ == "__main__":
    unittest.main()
