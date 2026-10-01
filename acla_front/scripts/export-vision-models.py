"""One-time Ultralytics export; packaged inference runs entirely in ONNX Runtime Web."""
import json
import os
from pathlib import Path
import shutil
from importlib.metadata import distribution

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.venv' / 'track-vision-models'
TARGET = ROOT / 'public' / 'vision-models'
CACHE.mkdir(parents=True, exist_ok=True)
TARGET.mkdir(parents=True, exist_ok=True)
(CACHE / 'config').mkdir(exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(CACHE / 'config'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

from ultralytics import YOLO
import onnx

INPUT_SIZE = 768

shutil.copyfile(distribution('ultralytics').locate_file('ultralytics-8.4.154.dist-info/licenses/LICENSE'),
                TARGET / 'ULTRALYTICS-LICENSE.txt')

os.chdir(CACHE)
for task, checkpoint in [('depth', 'yolo26m-depth')]:
    destination = TARGET / f'{checkpoint}.onnx'
    if destination.exists() and destination.with_suffix('.json').exists():
        graph = onnx.load(str(destination), load_external_data=False)
        shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
        if shape == [1, 3, INPUT_SIZE, INPUT_SIZE]:
            print(f'Using cached {checkpoint}', flush=True)
            continue
    model = YOLO(f'{checkpoint}.pt', task=task)
    exported = Path(model.export(format='onnx', imgsz=INPUT_SIZE, batch=1, dynamic=False,
                                 half=False, simplify=False, opset=17, nms=False, device='cpu'))
    shutil.copyfile(exported, destination)
    destination.with_suffix('.json').write_text(json.dumps({'task': task, 'names': model.names, 'imgsz': INPUT_SIZE}, indent=2))
    print(f'Prepared {task}: {destination}', flush=True)
