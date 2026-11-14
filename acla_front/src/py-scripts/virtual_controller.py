"""JSON-lines controller bridge, with input-only discovery independent of vJoy."""
import argparse
import json
import math
import queue
import sys
import threading

from virtual_wheel import VJoyController
from physical_controller import PhysicalControllers


def valid_action(action):
    axes = {"throttle", "brake", "steering"}
    buttons = {"shiftUp", "shiftDown"}
    return (isinstance(action, dict) and axes <= set(action) <= axes | buttons
            and all(type(action[key]) in (int, float) and math.isfinite(action[key])
                    and (-1 if key == "steering" else 0) <= action[key] <= 1 for key in axes)
            and all(type(action.get(key, False)) is bool for key in buttons)
            and not (action.get("shiftUp", False) and action.get("shiftDown", False)))


def emit(message):
    print(json.dumps(message), flush=True)


def serve(controller, commands, emit_message=emit, physical=None):
    try:
        if controller is not None:
            controller.reset()
        emit_message({"type": "ready"})
        while True:
            try:
                command = commands.get(timeout=0.05)
            except queue.Empty:
                continue
            if command is None:
                break
            if isinstance(command, dict) and command.get("type") == "read-physical":
                if physical is None:
                    physical = PhysicalControllers()
                emit_message({"type": "physical", "id": command.get("id"), "devices": physical.read()})
                continue
            action = command.get("action") if isinstance(command, dict) else None
            if controller is None:
                raise ValueError("Physical controller detection cannot apply driving inputs.")
            if not valid_action(action):
                raise ValueError("Invalid controller action. Inputs released.")
            controller.apply(action)
            emit_message({"type": "applied", "id": command.get("id")})
    finally:
        try:
            if controller is not None:
                controller.reset()
        finally:
            try:
                if controller is not None:
                    controller.close()
            finally:
                if physical is not None:
                    physical.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--physical-only", action="store_true")
    args = parser.parse_args()
    try:
        controller = None if args.physical_only else VJoyController(args.device_id)
        physical = PhysicalControllers() if args.physical_only else None
    except Exception as error:
        message = ("Physical controller detection unavailable. " if args.physical_only else
                   "Virtual wheel unavailable. Install vJoy and enable X, Y and Z axes and at least 2 buttons on device "
                   f"{args.device_id} in Configure vJoy. See README for setup. ")
        emit({"type": "error", "error": message + str(error)})
        return 1
    commands = queue.Queue(maxsize=8)

    def read_commands():
        try:
            for line in sys.stdin:
                commands.put(json.loads(line))
        except (ValueError, OSError):
            commands.put({})
        finally:
            commands.put(None)

    threading.Thread(target=read_commands, daemon=True).start()
    try:
        serve(controller, commands, physical=physical)
    except Exception as error:
        emit({"type": "error", "error": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
