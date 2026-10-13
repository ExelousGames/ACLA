# Ultralytics model storage

Backend storage for Ultralytics `.pt` weights trained with LabelMe annotations.
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

- `file`: a non-empty `.pt` file, up to 512 MiB.
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
`segment`, `classify`, `pose`, and `obb`. `classNames` must contain unique,
non-empty strings **in model class-ID order**: array index 0 is class 0.
The nested `metadata` object is optional and preserves training parameters,
metrics, dataset details, and run identifiers supplied by the uploader.

Example request (shell syntax; save the JSON above as `model-metadata.json`):

```bash
curl -X POST http://localhost:7001/ai-model/ultralytics \
  -H "Authorization: Bearer $BACKEND_TOKEN" \
  -F "file=@best.pt" \
  -F "metadata=<model-metadata.json"
```

The response includes `_id`, the supplied metadata, `framework: "ultralytics"`,
`annotationFormat: "labelme"`, `modelFileId`, `filename`, `sizeBytes`, `sha256`,
`createdAt`, and `updatedAt`. Use `_id` in the read/download routes. Each upload
creates a new record, including uploads with the same name; it does not replace
or activate an existing model.

Uploads use temporary disk storage, then stream to GridFS while computing the
SHA-256 checksum. Temporary files are removed after success or failure. If saving
the metadata fails, the uploaded GridFS file is removed. The GridFS write phase
has a 60-second timeout and aborts incomplete writes. Downloads stream binary
bytes rather than JSON or base64.

Invalid metadata, missing/empty files, unsupported extensions, and invalid IDs
return 400. Oversized files return 413. Missing records or files return 404.
The backend stores the artifact as supplied; it does not execute or validate the
checkpoint contents. Uploading from the training service and runtime model
selection remain separate integration work.

## Backend verification

```bash
npm test -- --runInBand ultralytics-model gridfs.service.spec
npm run build
```

Tests exercise the backend module over HTTP, with MongoDB/GridFS mocked and JWT
verification replaced by a test guard. They require neither the training service
nor a running database.
