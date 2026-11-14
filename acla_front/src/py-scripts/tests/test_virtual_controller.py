import queue
import os
import sys
import time
import unittest
from io import StringIO
from unittest.mock import Mock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from virtual_controller import main, serve, valid_action
from virtual_wheel import AXES, VJoyController


class ControllerTests(unittest.TestCase):
    def test_physical_only_startup_never_opens_vjoy(self):
        with patch.object(sys, "argv", ["virtual_controller.py", "--physical-only"]), \
                patch.object(sys, "stdin", StringIO('{"id": 1, "type": "read-physical"}\n')), \
                patch("virtual_controller.VJoyController") as wheel, \
                patch("virtual_controller.PhysicalControllers") as physical, \
                patch.object(sys, "stdout", StringIO()) as output:
            physical.return_value.read.return_value = []
            self.assertEqual(main(), 0)
            wheel.assert_not_called()
            physical.return_value.read.assert_called_once()
            physical.return_value.close.assert_called_once()
            self.assertEqual(output.getvalue().splitlines(), ['{"type": "ready"}',
                '{"type": "physical", "id": 1, "devices": []}'])

    def test_physical_only_bridge_rejects_output_and_closes_on_read_failure(self):
        for command in [{"action": {"throttle": 0, "brake": 0, "steering": 0}}, {"type": "read-physical"}]:
            physical, commands = Mock(), queue.Queue()
            physical.read.side_effect = RuntimeError('Input disconnected')
            commands.put(command)
            with self.assertRaises((ValueError, RuntimeError)):
                serve(None, commands, lambda _: None, physical=physical)
            physical.close.assert_called_once()

    def test_reports_physical_setup_failure_without_requiring_vjoy(self):
        with patch.object(sys, "argv", ["virtual_controller.py", "--physical-only"]), \
                patch("virtual_controller.PhysicalControllers", side_effect=RuntimeError('SDL missing')), \
                patch("virtual_controller.VJoyController") as wheel, patch("virtual_controller.emit") as emit:
            self.assertEqual(main(), 1)
            wheel.assert_not_called()
            self.assertEqual(emit.call_args.args[0], {
                'type': 'error', 'error': 'Physical controller detection unavailable. SDL missing'})

    def test_physical_reads_do_not_change_output(self):
        controller, physical = Mock(), Mock()
        physical.read.return_value = [{"id": "wheel", "axes": {"axis0": 0.5}, "buttons": []}]
        commands, messages = queue.Queue(), []
        commands.put({"id": 1, "action": {"throttle": 0.8, "brake": 0, "steering": 0}})
        commands.put({"id": 2, "type": "read-physical"})
        commands.put(None)
        serve(controller, commands, messages.append, physical=physical)
        self.assertIn({"type": "physical", "id": 2, "devices": physical.read.return_value}, messages)
        controller.apply.assert_called_once()
        self.assertEqual(controller.reset.call_count, 2)
        controller.close.assert_called_once()
        physical.close.assert_called_once()

    def test_physical_read_failure_releases_output_and_input_handles(self):
        controller, physical, commands = Mock(), Mock(), queue.Queue()
        physical.read.side_effect = RuntimeError('Controller input failed')
        commands.put({'type': 'read-physical', 'id': 1})
        with self.assertRaisesRegex(RuntimeError, 'Controller input failed'):
            serve(controller, commands, lambda _: None, physical=physical)
        self.assertEqual(controller.reset.call_count, 2)
        controller.close.assert_called_once()
        physical.close.assert_called_once()

    def test_applies_controls_and_releases_on_eof(self):
        controller = Mock()
        commands = queue.Queue()
        commands.put({"id": 1, "action": {"throttle": 0.4, "brake": 0.1, "steering": -0.5}})
        commands.put(None)
        messages = []
        serve(controller, commands, messages.append)
        self.assertEqual(controller.mock_calls, [call.reset(),
            call.apply({"throttle": 0.4, "brake": 0.1, "steering": -0.5}), call.reset(), call.close()])
        self.assertEqual(messages[-1], {"type": "applied", "id": 1})

    def test_training_pause_keeps_controller_connected_until_next_action(self):
        controller = Mock()
        first = {"throttle": 0.8, "brake": 0, "steering": 0.2}
        second = {"throttle": 0.3, "brake": 0, "steering": -0.1}

        class PausedQueue:
            polls = 0

            def get(self, timeout):
                self.polls += 1
                if self.polls == 1:
                    return {"id": 1, "action": first}
                if self.polls <= 16:
                    controller.close.assert_not_called()
                    controller.apply.assert_called_once_with(first)
                    # Fifteen empty 50 ms polls exceed the former 600 ms cutoff.
                    time.sleep(timeout)
                    raise queue.Empty
                if self.polls == 17:
                    return {"id": 2, "action": second}
                return None

        messages = []
        serve(controller, PausedQueue(), messages.append)
        self.assertEqual(controller.reset.call_count, 2)
        self.assertEqual(controller.apply.call_args_list, [call(first), call(second)])
        self.assertEqual(messages, [{"type": "ready"}, {"type": "applied", "id": 1}, {"type": "applied", "id": 2}])
        controller.close.assert_called_once()

    def test_invalid_commands_reset_device(self):
        for value in [float("nan"), float("inf"), -0.1, 2, True, "0.5"]:
            action = {"throttle": value, "brake": 0, "steering": 0}
            self.assertFalse(valid_action(action))
            commands = queue.Queue()
            commands.put({"action": action})
            controller = Mock()
            with self.assertRaises(ValueError):
                serve(controller, commands, lambda _: None)
            self.assertEqual(controller.reset.call_count, 2)
            controller.apply.assert_not_called()
            controller.close.assert_called_once()

    def test_shift_commands_are_boolean_and_mutually_exclusive(self):
        neutral = {"throttle": 0, "brake": 0, "steering": 0}
        for buttons in [{}, {"shiftUp": True}, {"shiftDown": True}, {"shiftUp": False, "shiftDown": False}]:
            self.assertTrue(valid_action({**neutral, **buttons}))
        for buttons in [{"shiftUp": True, "shiftDown": True}, {"shiftUp": 1}, {"shiftDown": 0},
                        {"shiftUp": None}, {"shiftDown": "true"}, {"unknown": False}]:
            commands, controller = queue.Queue(), Mock()
            commands.put({"action": {**neutral, **buttons}})
            with self.assertRaises(ValueError):
                serve(controller, commands, lambda _: None)
            controller.apply.assert_not_called()
            self.assertEqual(controller.reset.call_count, 2)
            controller.close.assert_called_once()

    def test_starts_wheel_by_default_and_with_selected_device(self):
        for args, device_id in [([], 1), (["--device-id", "4"], 4)]:
            with self.subTest(args=args), patch.object(sys, "argv", ["virtual_controller.py", *args]), \
                    patch.object(sys, "stdin", StringIO()), patch("virtual_controller.VJoyController") as wheel, \
                    patch.object(sys, "stdout", StringIO()):
                self.assertEqual(main(), 0)
                wheel.assert_called_once_with(device_id)
                self.assertEqual(wheel.return_value.reset.call_count, 2)
                wheel.return_value.close.assert_called_once()

    def test_rejects_invalid_device_ids_before_creating_device(self):
        for device_id in ["0", "17", "1.5", "invalid"]:
            with self.subTest(device_id=device_id), \
                    patch.object(sys, "argv", ["virtual_controller.py", "--device-id", device_id]), \
                    patch.object(sys, "stderr", StringIO()), patch("virtual_controller.VJoyController") as wheel:
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                wheel.assert_not_called()

    def test_reports_wheel_setup_failure(self):
        with patch.object(sys, "argv", ["virtual_controller.py"]), \
                patch("virtual_controller.VJoyController", side_effect=RuntimeError("Driver missing")), \
                patch("virtual_controller.emit") as emit:
            self.assertEqual(main(), 1)
            self.assertEqual(emit.call_args.args[0]["type"], "error")
            self.assertIn("Install vJoy", emit.call_args.args[0]["error"])


def wheel_sdk(low=0, high=32768):
    sdk = Mock()
    sdk.GetVJDStatus.return_value = 1
    sdk.GetVJDButtonNumber.return_value = 2

    def limit(value):
        def write(_device, _axis, pointer):
            pointer._obj.value = value
            return True
        return write

    sdk.GetVJDAxisMin.side_effect = limit(low)
    sdk.GetVJDAxisMax.side_effect = limit(high)
    return sdk


class WheelControllerTests(unittest.TestCase):
    def test_shift_paddles_use_distinct_buttons_and_axis_only_commands_release_them(self):
        sdk = wheel_sdk()
        controller = VJoyController(3, sdk)
        neutral = {"steering": 0, "throttle": 0, "brake": 0}
        controller.apply({**neutral, "shiftDown": True})
        controller.apply({**neutral, "shiftUp": True})
        controller.apply(neutral)
        self.assertEqual(sdk.SetBtn.call_args_list, [
            call(False, 3, 1), call(True, 3, 2),
            call(False, 3, 2), call(True, 3, 1),
            call(False, 3, 1), call(False, 3, 2),
        ])

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
        commands.put({"id": 12, "action": {"steering": 0.5, "throttle": 1, "brake": 0, "shiftUp": True}})
        commands.put(None)
        messages = []

        def receive(message):
            if message["type"] == "ready":
                self.assertEqual(sdk.SetAxis.call_count, 3)
                self.assertEqual(sdk.SetBtn.call_args_list, [call(False, 1, 1), call(False, 1, 2)])
            messages.append(message)

        serve(VJoyController(1, sdk), commands, receive)
        self.assertEqual(messages, [{"type": "ready"}, {"type": "applied", "id": 12}])
        self.assertEqual(sdk.mock_calls[-6:], [call.SetAxis(16384, 1, 0x30), call.SetAxis(0, 1, 0x31),
                                             call.SetAxis(0, 1, 0x32), call.SetBtn(False, 1, 1),
                                             call.SetBtn(False, 1, 2), call.RelinquishVJD(1)])

    def test_eof_neutralizes_and_relinquishes_wheel_after_idle(self):
        class StalledQueue:
            polls = 0

            def get(self, timeout):
                self.polls += 1
                if self.polls == 1:
                    return {"action": {"steering": 0, "throttle": 0, "brake": 0, "shiftDown": True}}
                if self.polls == 10:
                    return None
                raise queue.Empty

        sdk, commands = wheel_sdk(), StalledQueue()
        serve(VJoyController(1, sdk), commands, lambda _: None)
        self.assertEqual(sdk.mock_calls[-6:], [call.SetAxis(16384, 1, 0x30), call.SetAxis(0, 1, 0x31),
                                             call.SetAxis(0, 1, 0x32), call.SetBtn(False, 1, 1),
                                             call.SetBtn(False, 1, 2), call.RelinquishVJD(1)])

    def test_driver_and_axis_failures_do_not_claim_device(self):
        for method, value, message in [("vJoyEnabled", False, "disabled"), ("GetVJDStatus", 2, "busy"),
                                       ("GetVJDAxisExist", False, "Enable X"), ("AcquireVJD", False, "acquire"),
                                       ("GetVJDButtonNumber", 0, "at least 2 buttons"),
                                       ("GetVJDButtonNumber", 1, "at least 2 buttons")]:
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
        self.assertEqual(sdk.mock_calls[-5:], [call.SetAxis(0, 1, 0x31), call.SetAxis(0, 1, 0x32),
                                             call.SetBtn(False, 1, 1), call.SetBtn(False, 1, 2), call.RelinquishVJD(1)])

    def test_failed_button_write_releases_other_controls_and_is_not_acknowledged(self):
        sdk = wheel_sdk()
        sdk.SetBtn.side_effect = [True, True, True, False, False, True]
        commands = queue.Queue()
        commands.put({"id": 3, "action": {"steering": 0, "throttle": 0, "brake": 0, "shiftUp": True}})
        messages = []
        with self.assertRaisesRegex(RuntimeError, "stopped accepting shiftUp"):
            serve(VJoyController(1, sdk), commands, messages.append)
        self.assertEqual(messages, [{"type": "ready"}])
        self.assertEqual(sdk.mock_calls[-6:], [call.SetAxis(16384, 1, 0x30), call.SetAxis(0, 1, 0x31),
                                             call.SetAxis(0, 1, 0x32), call.SetBtn(False, 1, 1),
                                             call.SetBtn(False, 1, 2), call.RelinquishVJD(1)])


if __name__ == "__main__":
    unittest.main()
