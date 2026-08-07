# Replacement API contract

This contract is frozen for the replacement `/v1` API. During migration the
existing unversioned routes and their response models remain callable. Additive
fields may be introduced within v1; removing or changing a field requires a new
API version. Legacy routes are removed only in the final cutover task.

## Inputs

All files are multipart uploads.

| Endpoint | Multipart fields |
|---|---|
| `POST /v1/ocr/passport` | one `image` |
| `POST /v1/ocr/id-card` | one `front` and one `back` |
| `POST /v1/ocr/driving-license` | one `image` |
| `POST /v1/ocr/passport/batch` | `images` and/or one supported `archive` |
| `POST /v1/ocr/id-card/batch` | one `archive`; each immediate directory is one card containing `front` and `back` images |
| `POST /v1/ocr/driving-license/batch` | `images` and/or one supported `archive` |

Batch outputs preserve input order. One bad item produces an item-level error
and does not fail successful siblings. Request-wide validation errors use
`OcrErrorResponse`. Single successes use `OcrResponse`; batch successes use
`OcrBatchResponse`. Their typed definitions are in `app.api.schemas` and
`app.contracts`.

Confidence is always `{score, source, calibrated_probability}`. OCR and
detection scores are source-labelled and default to
`calibrated_probability: false`. Field boxes are normalized to `[0, 1]` within
their owning profile region. MRZ and cross-field checks are validation records,
not confidence scores.

## Error codes

| Code | Meaning |
|---|---|
| `invalid_upload` | Missing, empty, duplicated, or undecodable upload |
| `unsupported_media_type` | File type is not accepted |
| `upload_too_large` | File or expanded archive exceeds its configured limit |
| `too_many_items` | Expanded batch exceeds its configured item limit |
| `invalid_archive` | ZIP structure, path, or compression metadata is invalid |
| `missing_document_side` | An ID-card pair lacks `front` or `back` |
| `invalid_document` | The image cannot be localized or matched to the selected layout |
| `profile_unavailable` | Required profile is missing or invalid |
| `model_unavailable` | Required pinned model cannot be loaded |
| `queue_full` | Bounded processing queue has no capacity |
| `processing_failed` | Unexpected item-level processing failure |

Client input errors map to HTTP 4xx responses, unavailable runtime resources to
503, and unexpected processing failures to 500. Readiness returns non-success
while a required profile or model is unavailable.

## ZIP batches and limits

ZIP archives are read in memory and are never extracted to the filesystem.
Metadata entries (`__MACOSX`, `.DS_Store`, `Thumbs.db`) are ignored; encrypted,
absolute, and traversal paths are rejected. `BATCH_MAX_FILE_BYTES` limits each
upload and entry, `BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES` limits the expanded
archive, and `BATCH_MAX_FILES` limits accepted logical items.

An ID-card ZIP may nest card directories. Each directory contains exactly one
supported `front.jpg`/`front.jpeg`/`front.png` and one corresponding `back.*`;
directory membership and those filenames are the only pairing rule.

Examples:

```bash
curl -F image=@passport.jpg http://localhost:8000/v1/ocr/passport
curl -F images=@passport-1.jpg -F images=@passport-2.jpg http://localhost:8000/v1/ocr/passport/batch
curl -F front=@front.jpg -F back=@back.jpg http://localhost:8000/v1/ocr/id-card
curl -F archive=@id_cards.zip http://localhost:8000/v1/ocr/id-card/batch
```
