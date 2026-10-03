"""Prepare Depth Anything V2 Small; packaged inference uses ONNX Runtime Web."""
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.venv' / 'track-vision-models' / 'huggingface'
TARGET = ROOT / 'public' / 'vision-models' / 'depth-anything-v2-small.onnx'
MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
REVISION = '5426e4f0f36572d16453bbda7a8389317b1bef99'
INPUT_SIZE = 518
INPUT_SIZES = (252, 392, INPUT_SIZE)


def validate_export(path, input_size):
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    inputs, outputs = graph.graph.input, graph.graph.output
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    if (len(inputs) != 1 or len(outputs) != 1
            or shape(inputs[0]) != [1, 3, input_size, input_size]
            or shape(outputs[0]) != [1, input_size, input_size]
            or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
        raise ValueError(f'Expected float32 Depth Anything input [1, 3, {input_size}, {input_size}] and output [1, {input_size}, {input_size}].')


def main():
    missing = []
    for input_size in INPUT_SIZES:
        target = TARGET if input_size == INPUT_SIZE else TARGET.with_name(f'{TARGET.stem}-{input_size}.onnx')
        if target.exists():
            try:
                validate_export(target, input_size)
                print(f'Using cached Depth-Anything-V2-Small: {target}', flush=True)
                continue
            except Exception as error:
                print(f'Rebuilding invalid depth export: {error}', flush=True)
        missing.append((input_size, target))
    if not missing:
        return

    import torch
    from transformers import DepthAnythingForDepthEstimation

    class DepthExport(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).predicted_depth

    CACHE.mkdir(parents=True, exist_ok=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    model = DepthExport(DepthAnythingForDepthEstimation.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=str(CACHE), attn_implementation='eager',
    )).eval()
    for input_size, target in missing:
        temporary = target.with_suffix('.tmp')
        with torch.no_grad():
            torch.onnx.export(model, torch.zeros(1, 3, input_size, input_size), str(temporary),
                              input_names=['pixel_values'], output_names=['predicted_depth'],
                              opset_version=17, dynamo=False)
        validate_export(temporary, input_size)
        temporary.replace(target)
        print(f'Prepared Depth-Anything-V2-Small ({input_size}px): {target}', flush=True)


if __name__ == '__main__':
    main()
