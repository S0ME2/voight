# API

The maintained API is versioned under `/v1`. Successful single requests return
`{"result": ...}`. Batch requests preserve input order and give every item
exactly one result or error.

## Health

```bash
curl --fail http://127.0.0.1:8000/v1/health/live
curl --fail http://127.0.0.1:8000/v1/health/ready
```

Readiness initializes and validates configured models. With `PRELOAD=false`,
the first readiness call can take longer than later calls.

## Single documents

```bash
# Passport
curl --fail -F image=@passport.jpg \
  http://127.0.0.1:8000/v1/ocr/passport

# One logical ID card: both sides are required
curl --fail -F front=@front.jpg -F back=@back.jpg \
  http://127.0.0.1:8000/v1/ocr/id-card

# Driving licence
curl --fail -F image=@driving-license.jpg \
  http://127.0.0.1:8000/v1/ocr/driving-license
```

## Batches

Passport and driving-licence batches accept repeated `images` fields, one ZIP
`archive`, or both:

```bash
curl --fail -F images=@passport-1.jpg -F images=@passport-2.jpg \
  http://127.0.0.1:8000/v1/ocr/passport/batch

curl --fail -F archive=@driving-licences.zip \
  http://127.0.0.1:8000/v1/ocr/driving-license/batch
```

An ID-card batch is one ZIP with one directory per logical card:

```text
id-cards.zip
├── card-001/front.jpg
├── card-001/back.jpg
├── card-002/front.png
└── card-002/back.png
```

```bash
curl --fail -F archive=@id-cards.zip \
  http://127.0.0.1:8000/v1/ocr/id-card/batch
```

ZIP entries are read in memory. Absolute/traversal paths, encrypted entries,
oversized entries, and malformed side pairs are rejected. Upload limits are
documented in [configuration.md](configuration.md). The live OpenAPI schema and
interactive examples are served at `/openapi.json` and `/docs`.
