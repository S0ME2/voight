# Legacy API migration

The unversioned migration endpoints were removed after the `/v1` CPU contracts,
real batching, error isolation, and document behavior passed their replacement
tests.

| Removed endpoint | Replacement |
|---|---|
| `POST /ocr/passport/mrz` | `POST /v1/ocr/passport` |
| `POST /ocr/id-card/mrz` | `POST /v1/ocr/id-card` with `front` and `back` |
| `POST /ocr/driving-license/extract` | `POST /v1/ocr/driving-license` |
| unversioned `/batch` variants | matching `/v1/ocr/.../batch` route |
| `POST /ocr/file-type` | removed; submit to the intended typed route |

The v1 API returns structured document, confidence, validation, timing, and
per-item error envelopes instead of plain MRZ text or the legacy batch schema.
Clients must update both multipart field names and response parsing. See
[api.md](api.md) for exact commands.
