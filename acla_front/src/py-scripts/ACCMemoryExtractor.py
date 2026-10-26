from pyaccsharedmemory import accSharedMemory
from acc_broadcasting import ACCBroadcasting
import sched, time
import json
import os
import sys
import enum

recordedData = []


def serialize_shared_memory(value):
    """Encode SDK objects for transport without flattening or cleaning fields."""
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, '__dict__'):
        return {key: item for key, item in vars(value).items() if not key.startswith('_')}
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


class ACCRecording:
    def __init__(self, target_fps: int = 60):
        self.asm = accSharedMemory()
        self.broadcasting = ACCBroadcasting()
        self.target_fps = target_fps if target_fps > 0 else 60
        self._frame_delay = 1.0 / float(self.target_fps)
        return

    def startRecording(self,full_path):
        my_scheduler = sched.scheduler(time.time, time.sleep)
        # 60 FPS = 1/60 seconds delay between calls
        my_scheduler.enter(self._frame_delay, 1, self.recordOnce, (my_scheduler,full_path))
        try:
            my_scheduler.run()
        finally:
            self.broadcasting.close()

    def recordOnce(self,scheduler,full_path): 
        # schedule the next call first
        scheduler.enter(self._frame_delay, 1, self.recordOnce, (scheduler,full_path))

        # Broadcasting belongs to the reader, even while shared memory has no fresh sample.
        positions = self.broadcasting.poll()
        sm = self.asm.read_shared_memory()
        if  (sm is not None):
            packet = {
                'type': 'sample',
                'sharedMemory': sm,
                'broadcasting': {
                    'positions': positions,
                    'sessionTime': self.broadcasting.get_session_time(),
                },
            }
            # The Electron ACC adapter owns all application field conversions.
            print(json.dumps(packet, default=serialize_shared_memory, separators=(',', ':')), flush=True)
        else:
            print(json.dumps({"available": False}).rstrip(), flush=True)
            

if __name__ == "__main__":
    recorder = ACCRecording()
    # The Electron recording reader uses this script as a stream-only telemetry
    # source. Legacy positional output arguments remain accepted for unrelated
    # callers, but this process never owns persistence.
    if "--stream" in sys.argv[1:]:
        recorder.startRecording(None)
    elif len(sys.argv) >= 3:
        folder = sys.argv[1]
        filename = sys.argv[2]
        full_path = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        recorder.startRecording(full_path)
    else:
        raise SystemExit("Use --stream or provide an output folder and filename")
