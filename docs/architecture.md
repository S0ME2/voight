# Architecture

## Request flow

```text
multipart upload or ZIP
→ validate and decode logical documents
→ batch document/MRZ localization
→ canonical perspective transform
→ profile data crop and field ROIs
→ batched Paddle text detection
→ discard visible lines outside field ROIs
→ batched line recognition
→ field assignment and document-specific parsing
→ MRZ and cross-field validation
→ ordered response or isolated item error
```

`app/main.py` is the composition root. `app/api/v1.py` owns HTTP transport,
safe in-memory ZIP expansion, asynchronous admission, and conversion to the
public schemas in `app/contracts.py` and `app/api/schemas.py`.

`app/inference/batch.py` is the shared coordinator. It groups compatible work
so DocAligner, MRZScanner, Paddle detection, and Paddle recognition receive
real tensors containing more than one sample when a request permits it. It
restores output ordering and isolates recoverable item failures. Separate HTTP
requests are admitted asynchronously but are not combined into one model call.

## Model roles

- **DocAligner** localizes and rectifies driving licences and ID-card sides.
- **MRZScanner detection** localizes the passport data page through its MRZ and
  probes the ID-card back; the front is probed only when the back has no usable
  MRZ polygon.
- **PaddleOCR detector** finds visible and MRZ text lines after rectification.
- **PaddleOCR recognizer** is the default recognizer for visible text and MRZ.
- **MRZScanner recognition** is an optional specialized adapter. It remains
  experimental and is not the default.

`app/models.py` owns the one heavyweight instance of each selected model.
Project-owned contracts in `app/inference/contracts.py` keep document pipelines
independent of third-party result shapes. Backend and model names come from
validated environment settings, so supported models can be swapped without
changing pipeline orchestration.

## Document behavior

Production profile JSON lives in `config/documents/`. Passport and ID-card
pipelines in `app/documents/identity.py` own their regions, MRZ side/page, and
visible-to-MRZ reconciliation. Driving-licence parsing remains in
`app/documents/driving_license_fields.py`. Geometry, ROI assignment, confidence
aggregation, and artifact handling are shared.

OCR scores are returned with their source and
`calibrated_probability: false`; they are not represented as probabilities.
