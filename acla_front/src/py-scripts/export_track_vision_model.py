"""Export a verified backend checkpoint for local Track Vision inference."""
import argparse
import json
import os
from pathlib import Path

# Loading/exporting must never fetch weights or install packages at runtime.
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
if os.environ.get('YOLO_CONFIG_DIR'):
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)


def export_model(weights: Path, labels: list[str], input_size: int = 768, output: Path | None = None) -> Path:
    if input_size not in (384, 640, 768):
        raise ValueError('Unsupported Track Vision input resolution.')
    weights = weights.resolve(strict=True)
    if weights.suffix != '.pt' or not weights.is_file():
        raise ValueError('Track Vision requires a local .pt checkpoint.')
    from ultralytics import YOLO

    model = YOLO(str(weights), task='segment')
    if model.task != 'segment':
        raise ValueError('The uploaded model must perform instance segmentation.')
    if len(model.names) != len(labels):
        raise ValueError('Backend labels do not match the number of model classes.')
    model.model.names = dict(enumerate(labels))
    # nms=None selects the raw one-to-many head, including for YOLO26 exports.
    exported = Path(model.export(format='onnx', imgsz=input_size, batch=1, dynamic=False,
                                 half=False, simplify=False, opset=17, nms=None, device='cpu'))
    import onnx

    graph = onnx.load(str(exported))
    onnx.checker.check_model(graph)
    inputs = [tuple(d.dim_value for d in value.type.tensor_type.shape.dim) for value in graph.graph.input]
    if inputs != [(1, 3, input_size, input_size)]:
        raise ValueError('The model input does not match the selected resolution.')
    outputs = [tuple(d.dim_value for d in output.type.tensor_type.shape.dim) for output in graph.graph.output]
    predictions = next((shape for shape in outputs if len(shape) == 3), ())
    prototypes = next((shape for shape in outputs if len(shape) == 4), ())
    if (len(outputs) != 2 or not predictions or not prototypes or predictions[0] != 1
            or prototypes[0] != 1 or predictions[1] != 4 + len(labels) + prototypes[1]):
        raise ValueError('The model does not export raw segmentation predictions and mask prototypes.')
    return exported.replace(output) if output is not None else exported


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=json.loads)
    parser.add_argument('--input-size', type=int, choices=(384, 640, 768), default=768)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    export_model(args.weights, args.labels, args.input_size, args.output)
