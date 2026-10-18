# Ultralytics model storage

Backend storage for Ultralytics `.pt` weights trained with LabelMe annotations,
and ACLA YOLO position-regression `.pt` checkpoints and `.onnx` exports.
`UltralyticsModelModule` stores model metadata in MongoDB's `ultralytics_models`
collection and the unchanged binary file in the `ultralytics_models` GridFS bucket
(`ultralytics_models.files` / `ultralytics_models.chunks`). Its endpoints live
under `/ai-model/ultralytics`.

The implementation lives in `src/modules/ai-model/ultralytics/`, with its own
controller, upload DTO and validation, service, schema, and storage bucket.
`AiModelModule` imports the Ultralytics storage module and registers its controller
before the generic `:id` routes. Additional model families can have independent
contracts beneath `/ai-model/<family>` and share the GridFS file-transfer utility.

All endpoints require `Authorization: Bearer <backend JWT>`.

| Method | Route                                       | Result                                                   |
| ------ | ------------------------------------------- | -------------------------------------------------------- |
| POST   | `/ai-model/ultralytics`                     | Save weights and metadata; return the saved record (201) |
| GET    | `/ai-model/ultralytics?name=track-segments` | List metadata, newest first; optional exact name filter  |
| GET    | `/ai-model/ultralytics/:id`                 | Read one model's metadata                                |
| GET    | `/ai-model/ultralytics/:id/file`            | Stream the original binary weights                       |

## Upload contract

Send `multipart/form-data` with exactly these fields:

- `file`: a non-empty `.pt` or `.onnx` file, up to 512 MiB (extensions are case-insensitive).
- `metadata`: a JSON-encoded object, up to 1 MiB, as shown below.

```json
{
  "name": "track-segments",
  "task": "segment",
  "classNames": ["straight", "corner"],
  "metadata": {
    "trainingRunId": "run-001",
    "baseModel": "yolo11n-seg.pt",
    "epochs": 50,
    "metrics": { "map50": 0.92 }
  }
}
```

`name`, `task`, and `classNames` are required. Supported tasks are `detect`,
`segment`, `classify`, `pose`, `obb`, and the ACLA extension `position`.
`classNames` must contain unique,
non-empty strings **in model class-ID order**: array index 0 is class 0.
The nested `metadata` object is optional and preserves training parameters,
metrics, dataset details, and run identifiers supplied by the uploader.

For `position`, `classNames` describes the embedded segmentation model; regression
output names, target definition, coordinate units, and validation metrics belong
in `metadata`. Current position checkpoints use `metadata.format: "acla-yolo-position-v3"`
and include game/map-name vocabularies in `metadata.categories`. Python `.pt`
checkpoints load with `YOLOPositionModel.load`, not the standard Ultralytics loader.
The `train_position.py` pipeline publishes `best.onnx` with `task: "position"`
and `metadata.artifactFormat: "onnx"`. The backend preserves `metadata.inference`
unchanged, including tensor names/shapes, preprocessing, and category/context order.
Clients can load the downloaded ONNX bytes directly into ONNX Runtime using that
contract: inputs `images`, `references`, and `context`; outputs `deltas`,
`predictions`, and `prototypes`.

The trainer exports `best.onnx.json` even with `--no-upload`. That sidecar contains
the upload metadata except for the user-selected top-level `name`. Publication
reads the equivalent manifest embedded in the ONNX file and adds `name` before
sending it in the `metadata` field; no separate JSON file upload is needed.
The `/ai-model/ultralytics/track-vision` selector only returns `segment` models.

Example request (shell syntax; save the JSON above as `model-metadata.json`):

```bash
curl -X POST http://localhost:7001/ai-model/ultralytics \
  -H "Authorization: Bearer $BACKEND_TOKEN" \
  -F "file=@best.pt" \
  -F "metadata=<model-metadata.json"
```

For a manual position upload, add `"name": "track-position"` to the top-level
object from `best.onnx.json`, save it as `model-metadata.json`, and use
`-F "file=@best.onnx"` in the request above.

The response includes `_id`, the supplied metadata, `framework: "ultralytics"`,
`annotationFormat` (`"position"` for position tasks, otherwise `"labelme"`),
`modelFileId`, `filename`, `sizeBytes`, `sha256`,
`createdAt`, and `updatedAt`. Use `_id` in the read/download routes. Each upload
creates a new record, including uploads with the same name; it does not replace
or activate an existing model.

Uploads use temporary disk storage, then stream to GridFS while computing the
SHA-256 checksum. Temporary files are removed after success or failure. If saving
the metadata fails, the uploaded GridFS file is removed. The GridFS write phase
has a 60-second timeout and aborts incomplete writes. Downloads stream binary
bytes rather than JSON or base64.
Download headers retain the uploaded filename and use a matching `.pt` or `.onnx`
extension in the fallback filename. The list and individual metadata routes
return the complete inference metadata alongside the filename and SHA-256 checksum.

Invalid metadata, missing/empty files, unsupported extensions, and invalid IDs
return 400. Oversized files return 413. Missing records or files return 404.
The backend stores the artifact as supplied; it does not execute or validate the
checkpoint contents. Runtime model selection and inference remain the client's
responsibility.

## Backend verification

```bash
npm test -- --runInBand ultralytics-model gridfs.service.spec
npm run build
```

Tests exercise the backend module over HTTP, with MongoDB/GridFS mocked and JWT
verification replaced by a test guard. They require neither the training service
nor a running database.
