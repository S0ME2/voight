# Application layout

`main.py` assembles settings, models, and the API router. `api/` handles HTTP uploads, batch expansion, response schemas, and request-level artifact organization. `workflows.py` owns operation selection, model acquisition, and calls into the document pipelines.

`models.py` is the only place where the three heavy models are constructed and retained. `PRELOAD=true` initializes all models during application startup. With `PRELOAD=false`, each required model is initialized on its first use and then reused.

`imaging.py`, `ocr.py`, and `roi.py` contain generic mechanics shared by pipelines. `documents/mrz.py` is the common MRZ localization and reconstruction implementation, configured by an ID-card or passport profile. `documents/driving_license.py` contains the alignment and ROI pipeline; `driving_license_fields.py` contains field-specific rules and parsing.

Use `python -m app.tools.roi_editor` to create crop or field ROI JSON and preview a configuration. Tools depend on production ROI mechanics, never the reverse.

## Existing single-file endpoints

The original endpoints remain available and keep their original response types:

- `POST /ocr/file-type`
- `POST /ocr/id-card/mrz`
- `POST /ocr/passport/mrz`
- `POST /ocr/driving-license/extract`

Single-file artifact runs keep the existing top-level layout under their operation folder.

## Batch endpoints

- `POST /ocr/id-card/mrz/batch`
- `POST /ocr/passport/mrz/batch`
- `POST /ocr/driving-license/extract/batch`

The multipart field is named `files`. It accepts:

- one or more image files;
- one or more ZIP archives;
- a mixture of direct images and ZIP archives.

ZIP archives may contain nested folders. Directory entries, `__MACOSX`, `.DS_Store`, and `Thumbs.db` are ignored. Other non-image files are returned as failed batch items instead of aborting successful images.

Batch processing is sequential because the heavy model objects are shared and are not assumed to be thread-safe. Upload order is preserved. ZIP entries preserve archive order. A failed image does not stop later images unless the process runs out of memory.

### Swagger behavior

FastAPI 0.129.1 and later changed `UploadFile` schemas to the OpenAPI 3.1
`contentMediaType` representation. Swagger UI 5.x currently mishandles that
representation for arrays and renders `array<string>` text boxes containing
garbled file data. The reusable upload types in `api/upload_types.py` override
only the documentation schema to include `format: binary` alongside the OpenAPI
3.1 media type; runtime values remain normal `UploadFile` objects. Remove this
isolated workaround once Swagger UI supports binary array items declared with
`contentMediaType`.

Swagger UI can select multiple individual files through the `files` field. Standard Swagger UI does not provide browser folder selection because OpenAPI file inputs do not expose the non-standard `webkitdirectory` attribute.

To submit a folder through Swagger, compress it as a ZIP and upload the ZIP. The batch routes expand it in memory and process its images.

### Batch artifact layout

With `LOGGING=true`, one parent directory is created per batch:

```text
logs/
└── passport_mrz_batch/
    └── 1_batch_12-30-45-123456_24-07-2026/
        ├── batch_metadata.json
        ├── batch_result.json
        ├── timing.json
        ├── 001_passport-front/
        │   ├── 00_input.png
        │   ├── 00_input_metadata.json
        │   ├── ... existing MRZ intermediate artifacts ...
        │   └── timing.json
        └── 002_passport-back/
            ├── 00_input.png
            ├── 00_input_metadata.json
            ├── error.json
            └── timing.json
```

Each image directory receives the same pipeline artifacts as a single-file run, plus a standardized `timing.json`. Failed images keep their input, metadata, error, and timing information.

The parent `timing.json` contains:

- route-handler wall-clock time for the complete batch;
- upload or ZIP preparation time;
- model loading time that occurred during the batch;
- previously recorded initialization times for models loaded at startup;
- total, successful, and failed item counts.

The child `timing.json` includes input preparation, image decoding, processing, total per-image wall time, model loading during that item, and the pipeline’s detailed stage timings.

With `PRELOAD=false`, first-use model initialization occurs inside the request and is included in the first affected image and whole-batch wall time. With `PRELOAD=true`, model loading occurs before the route begins; startup load durations are reported separately and are not falsely added to request wall time.

Network transfer time before FastAPI enters the route is not included. The final write of `timing.json` itself is also necessarily outside the measured value stored in that same file.

### Limits

- `BATCH_MAX_FILES`: maximum expanded files per request; default `20`.
- `BATCH_MAX_FILE_BYTES`: maximum bytes for one direct upload or ZIP entry; default `52428800` (50 MiB).
- `BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES`: maximum total uncompressed bytes per ZIP; default `524288000` (500 MiB).

Requests exceeding a limit return HTTP `413`. Invalid or encrypted ZIP archives return HTTP `422`.

### cURL examples

Multiple images:

```bash
curl -X POST http://localhost:8000/ocr/passport/mrz/batch \
  -F "files=@passport_1.jpg" \
  -F "files=@passport_2.jpg"
```

A ZIP archive:

```bash
curl -X POST http://localhost:8000/ocr/passport/mrz/batch \
  -F "files=@passports.zip;type=application/zip"
```
