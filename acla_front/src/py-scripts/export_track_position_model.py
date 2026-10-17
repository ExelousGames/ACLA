"""Export the published position add-on without importing the training workspace."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'
if os.environ.get('YOLO_CONFIG_DIR'):
    Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)

OUTPUT_NAMES = [f'{reference}_delta_{axis}' for reference in ('reference', 'centerline') for axis in 'xyz']


def load_position_model(weights: Path, labels: list[str], input_size: int):
    import torch
    from torch import nn
    from ultralytics.nn.tasks import SegmentationModel

    saved = torch.load(weights, map_location='cpu', weights_only=True)
    # A version label cannot establish compatibility. Inspect the actual model
    # contract and weights, including checkpoints with new or missing labels.
    if saved.get('output_names') != OUTPUT_NAMES:
        raise ValueError('Position checkpoint output_names must describe reference and centerline XYZ deltas.')
    if saved.get('target_definition') != 'reference_position - current_position':
        raise ValueError('Position checkpoint target_definition must be reference_position - current_position.')
    if saved.get('coordinate_units') != 'source_coordinates':
        raise ValueError('Position checkpoint coordinate_units must be source_coordinates.')
    if saved.get('imgsz') != input_size:
        raise ValueError('Position inference must use the training input resolution.')
    names = saved['names']
    if [names[index] for index in range(len(names))] != labels:
        raise ValueError('Backend labels do not match the position checkpoint.')

    class PositionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.yolo = SegmentationModel(deepcopy(saved['yolo_yaml']), verbose=False).float().eval()
            self.yolo.names = names
            self.feature_layers = list(self.yolo.model[-1].f)
            self.register_buffer('position_mean', torch.zeros(6))
            self.register_buffer('position_std', torch.ones(6))
            first = saved['state_dict']['head.0.weight']
            last = saved['state_dict']['head.2.weight']
            self.head = nn.Sequential(nn.Linear(first.shape[1], first.shape[0]), nn.ReLU(),
                                      nn.Linear(last.shape[1], last.shape[0]))
            with torch.no_grad():
                feature_width = torch.stack(self.yolo.predict(torch.zeros(1, 3, 64, 64), embed=self.feature_layers)).shape[1]
            self.context_size = first.shape[1] - feature_width - 6
            if self.context_size:
                categories = saved.get('categories', {})
                expected = 1 + sum(len(categories.get(field, [])) for field in ('game', 'map_name'))
                if self.context_size != expected:
                    raise ValueError(f'Position head needs {self.context_size} context values but saved categories describe {expected}.')

        def forward(self, images, references, context=None):
            # Same pooled segmentation features and reference standardization as training.
            features = torch.stack(self.yolo.predict(images, embed=self.feature_layers))
            positions = (references.flatten(1) - self.position_mean) / self.position_std
            inputs = (features, positions) if context is None else (features, positions, context)
            return self.head(torch.cat(inputs, dim=1)).reshape(1, 2, 3)

    model = PositionModel()
    model.load_state_dict(saved['state_dict'], strict=True)
    if not torch.isfinite(model.position_std).all() or not (model.position_std > 0).all():
        raise ValueError('Invalid position input standardization.')
    return model.eval().requires_grad_(False)


def export_model(weights: Path, labels: list[str], input_size: int, output: Path):
    import onnx
    import torch

    if input_size < 32 or input_size > 2048 or input_size % 32:
        raise ValueError('Unsupported position input resolution.')
    weights = weights.resolve(strict=True)
    if weights.suffix != '.pt' or not weights.is_file():
        raise ValueError('Position inference requires a local .pt checkpoint.')
    model = load_position_model(weights, labels, input_size)
    inputs = (torch.zeros(1, 3, input_size, input_size), torch.zeros(1, 2, 3))
    input_names = ['images', 'references']
    if model.context_size:
        inputs += (torch.zeros(1, model.context_size),)
        input_names.append('context')
    torch.onnx.export(model, inputs, str(output),
                      input_names=input_names, output_names=['deltas'],
                      opset_version=17, dynamic_axes=None, dynamo=False)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    shapes = lambda values: [tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim) for value in values]
    if shapes(graph.graph.input) != [tuple(value.shape) for value in inputs] or shapes(graph.graph.output) != [(1, 2, 3)]:
        raise ValueError('Unexpected position ONNX inputs or outputs.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=json.loads)
    parser.add_argument('--input-size', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export_model(args.weights, args.labels, args.input_size, args.output)
