"""JSON-lines vJoy steering wheel bridge with an independent dead-man timeout."""
import argparse
import json
import math
import queue
import sys
import threading
import time

from virtual_wheel import VJoyController

WATCHDOG_SECONDS = 0.6


def valid_action(action):
    return isinstance(action, dict) and set(action) == {"throttle", "brake", "steering"} and all(
        type(value) in (int, float) and math.isfinite(value)
        and (-1 if key == "steering" else 0) <= value <= 1 for key, value in action.items()
    )


def emit(message):
    print(json.dumps(message), flush=True)


def serve(controller, commands, emit_message=emit, clock=time.monotonic):
    armed = False
    last_command = clock()
    try:
        controller.reset()
        emit_message({"type": "ready"})
        while True:
            if armed and clock() - last_command > WATCHDOG_SECONDS:
                raise RuntimeError("Controller watchdog expired. Inputs released; reconnect to continue.")
            try:
                command = commands.get(timeout=0.05)
            except queue.Empty:
                continue
            if command is None:
                break
            action = command.get("action") if isinstance(command, dict) else None
            if not valid_action(action):
                raise ValueError("Invalid controller action. Inputs released.")
            controller.apply(action)
            last_command = clock()
            armed = any(action.values())
            emit_message({"type": "applied", "id": command.get("id")})
    finally:
        try:
            controller.reset()
        finally:
            controller.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("vjoy",), default="vjoy")
    parser.add_argument("--device-id", type=int, choices=range(1, 17), default=1)
    args = parser.parse_args()
    try:
        controller = VJoyController(args.device_id)
    except Exception as error:
        message = ("Virtual wheel unavailable. Install vJoy and enable X, Y and Z axes on device "
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
        serve(controller, commands)
    except Exception as error:
        emit({"type": "error", "error": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
