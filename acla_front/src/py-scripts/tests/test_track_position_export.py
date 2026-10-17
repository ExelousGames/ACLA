"""Exercise the real checkpoint loader and ONNX graph using a small local YOLO model."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_track_position_model import OUTPUT_NAMES, export_model, load_position_model

import numpy as np
import torch
from torch import nn
from ultralytics.nn.tasks import SegmentationModel


class PositionExportTests(unittest.TestCase):
    def test_checkpoint_and_onnx_preserve_visual_features_and_reference_standardization(self):
        self.verify_checkpoint()

    def test_contextual_checkpoint_exports_progress_and_category_features_without_version_gates(self):
        self.verify_checkpoint({'game': ['acc'], 'map_name': ['monza', 'nurburgring_24h']})

    def verify_checkpoint(self, categories=None):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        yolo = SegmentationModel({'nc': 2, 'backbone': [
            [-1, 1, 'Conv', [8, 3, 2]], [-1, 1, 'Conv', [16, 3, 2]], [-1, 1, 'Conv', [32, 3, 2]],
        ], 'head': [[[0, 1, 2], 1, 'Segment', [2, 8, 16]]]}, verbose=False).eval()
        layers = list(yolo.model[-1].f)
        image = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            features = torch.stack(yolo.predict(image, embed=layers))
        # The saved weights, rather than a version-specific constant, define head width.
        context = torch.tensor([[0.25, 1., 0., 1.]]) if categories else None
        context_width = context.shape[1] if context is not None else 0
        head = nn.Sequential(nn.Linear(features.shape[1] + 6 + context_width, 48), nn.ReLU(), nn.Linear(48, 6)).eval()
        mean = torch.arange(6, dtype=torch.float32)
        std = torch.arange(1, 7, dtype=torch.float32)
        references = torch.tensor([[[10., -20., 30.], [-40., 50., -60.]]])
        state = {**{f'yolo.{key}': value for key, value in yolo.state_dict().items()},
                 **{f'head.{key}': value for key, value in head.state_dict().items()},
                 'position_mean': mean, 'position_std': std}
        saved = {'format': 'acla-yolo-position-v1', 'imgsz': 64, 'yolo_yaml': deepcopy(yolo.yaml),
                 'names': {0: 'track', 1: 'car'}, 'state_dict': state, 'output_names': OUTPUT_NAMES,
                 'target_definition': 'reference_position - current_position', 'coordinate_units': 'source_coordinates'}
        if categories:
            saved['categories'] = categories
        inputs = (image, references) if context is None else (image, references, context)
        feeds = {'images': image.numpy(), 'references': references.numpy()}
        if context is not None:
            feeds['context'] = context.numpy()
        with tempfile.TemporaryDirectory(prefix='acla-position-export-') as folder:
            checkpoint, output = Path(folder) / 'weights.pt', Path(folder) / 'position.onnx'
            torch.save(saved, checkpoint)
            model = load_position_model(checkpoint, ['track', 'car'], 64)
            with torch.no_grad():
                parts = (features, (references.flatten(1) - mean) / std)
                if context is not None:
                    parts += (context,)
                expected = head(torch.cat(parts, dim=1)).reshape(1, 2, 3)
                torch.testing.assert_close(model(*inputs), expected)
            export_model(checkpoint, ['track', 'car'], 64, output)
            from onnx.reference import ReferenceEvaluator
            session = ReferenceEvaluator(str(output))
            result = session.run(None, feeds)[0]
            np.testing.assert_allclose(result, expected.numpy(), rtol=1e-4, atol=1e-5)
            changed = references + 4
            with torch.no_grad():
                changed_expected = model(image, changed, context).numpy()
            changed_result = session.run(None, {**feeds, 'references': changed.numpy()})[0]
            np.testing.assert_allclose(changed_result, changed_expected, rtol=1e-4, atol=1e-5)
            self.assertFalse(np.allclose(result, changed_result))
            if context is not None:
                changed_context = torch.tensor([[0.75, 1., 1., 0.]])
                with torch.no_grad():
                    context_expected = model(image, references, changed_context).numpy()
                context_result = session.run(None, {**feeds, 'context': changed_context.numpy()})[0]
                np.testing.assert_allclose(context_result, context_expected, rtol=1e-4, atol=1e-5)
                self.assertFalse(np.allclose(result, context_result))
            del session
            # Version text is deliberately unrelated to inference compatibility.
            for label in ('acla-yolo-position-v2', 'future-model-v999', None):
                with self.subTest(format=label):
                    if label is None:
                        saved.pop('format', None)
                    else:
                        saved['format'] = label
                    torch.save(saved, checkpoint)
                    restored = load_position_model(checkpoint, ['track', 'car'], 64)
                    with torch.no_grad():
                        torch.testing.assert_close(restored(*inputs), expected)
            export_model(checkpoint, ['track', 'car'], 64, output)
            unversioned_result = ReferenceEvaluator(str(output)).run(None, feeds)[0]
            np.testing.assert_allclose(unversioned_result, expected.numpy(), rtol=1e-4, atol=1e-5)
            with self.assertRaisesRegex(ValueError, 'training input resolution'):
                load_position_model(checkpoint, ['track', 'car'], 768)
            with self.assertRaisesRegex(ValueError, 'Backend labels'):
                load_position_model(checkpoint, ['car', 'track'], 64)


if __name__ == '__main__':
    unittest.main()
