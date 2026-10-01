"""Prepare Depth Anything V2 Small; packaged inference uses ONNX Runtime Web."""
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / '.venv' / 'track-vision-models' / 'huggingface'
TARGET = ROOT / 'public' / 'vision-models' / 'depth-anything-v2-small.onnx'
MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
REVISION = '5426e4f0f36572d16453bbda7a8389317b1bef99'
INPUT_SIZE = 518


def validate_export(path):
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    inputs, outputs = graph.graph.input, graph.graph.output
    shape = lambda value: [d.dim_value for d in value.type.tensor_type.shape.dim]
    if (len(inputs) != 1 or len(outputs) != 1
            or shape(inputs[0]) != [1, 3, INPUT_SIZE, INPUT_SIZE]
            or shape(outputs[0]) != [1, INPUT_SIZE, INPUT_SIZE]
            or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
            or outputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
        raise ValueError('Expected float32 Depth Anything input [1, 3, 518, 518] and output [1, 518, 518].')


def main():
    if TARGET.exists():
        try:
            validate_export(TARGET)
            print(f'Using cached Depth-Anything-V2-Small: {TARGET}', flush=True)
            return
        except Exception as error:
            print(f'Rebuilding invalid depth export: {error}', flush=True)

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
    temporary = TARGET.with_suffix('.tmp')
    with torch.no_grad():
        torch.onnx.export(model, torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE), str(temporary),
                          input_names=['pixel_values'], output_names=['predicted_depth'],
                          opset_version=17, dynamo=False)
    validate_export(temporary)
    temporary.replace(TARGET)
    print(f'Prepared Depth-Anything-V2-Small: {TARGET}', flush=True)


if __name__ == '__main__':
    main()
