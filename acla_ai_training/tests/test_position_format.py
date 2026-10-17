"""Version labels do not control checkpoint loading or publication."""
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics.nn.tasks import SegmentationModel
from training.image_segmentation.position_model import YOLOPositionModel
from training.image_segmentation.position_publication import upload_position_checkpoint


class PositionFormatTests(unittest.TestCase):
    def test_load_and_publish_ignore_version_labels(self):
        torch.set_num_threads(1)
        yolo = SegmentationModel({'nc': 2, 'backbone': [
            [-1, 1, 'Conv', [8, 3, 2]], [-1, 1, 'Conv', [16, 3, 2]], [-1, 1, 'Conv', [32, 3, 2]],
        ], 'head': [[[0, 1, 2], 1, 'Segment', [2, 8, 16]]]}, verbose=False).eval()
        yolo.names = {0: 'track', 1: 'car'}
        categories = {"game": ["acc"], "map_name": ["monza", "nurburgring_24h"]}
        original = YOLOPositionModel(yolo, imgsz=64, categories=categories).eval()
        image, references = torch.rand(1, 3, 64, 64), torch.rand(1, 2, 3)
        context = torch.tensor([[0.25, 1., 1., 0.]])
        with torch.no_grad():
            expected = original(image, references, context)
        with tempfile.TemporaryDirectory(prefix='acla-position-format-') as folder:
            checkpoint = Path(folder) / 'weights.pt'
            original.save(checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            for label in ('acla-yolo-position-v1', 'acla-yolo-position-v2', 'acla-yolo-position-v3', 'future-v999', None):
                with self.subTest(format=label):
                    if label is None:
                        saved.pop('format', None)
                    else:
                        saved['format'] = label
                    torch.save(saved, checkpoint)
                    with torch.no_grad():
                        restored = YOLOPositionModel.load(checkpoint)
                        self.assertEqual(restored.categories, categories)
                        torch.testing.assert_close(restored(image, references, context), expected)
                    with patch('training.image_segmentation.position_publication.upload_ultralytics_model',
                               new_callable=AsyncMock, return_value={'_id': 'published'}) as upload:
                        result = asyncio.run(upload_position_checkpoint(checkpoint, backend_service=object()))
                        self.assertEqual(result, {'_id': 'published'})
                        metadata = upload.call_args.args[3]
                        self.assertEqual(metadata['metadata']['format'], label)
                        self.assertEqual(metadata['classNames'], ['track', 'car'])
                        self.assertEqual(metadata['metadata']['categories'], saved['categories'])


if __name__ == '__main__':
    unittest.main()
