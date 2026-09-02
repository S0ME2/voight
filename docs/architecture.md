# Architecture

## System map

```mermaid
flowchart LR
    client["Client"] --> api{"Route family"}
    api --> health["Health\n/v1/health/*"]
    api --> profile["Profile OCR\n/v1/ocr/*"]
    api --> verification["Verification OCR\n/verification/*/ocr"]
    api --> checks["Field checks\n/verification/*/check"]

    profile --> gate["Admission gate\nqueue + one active runner"]
    verification --> gate
    gate --> models["Shared process-owned models"]
    profile --> profile_pipeline["Document profiles\nlocalization · ROIs · parsing"]
    models --> profile_pipeline
    models --> verification_pipeline["Whole-image OCR\ntext detector · Latin recognizer"]
    verification --> verification_pipeline
    checks --> matcher["Model-free matcher\nnormalization · assignment · evidence"]

    classDef api fill:#172554,stroke:#60a5fa,color:#fff
    classDef engine fill:#3f1d5b,stroke:#c084fc,color:#fff
    classDef output fill:#064e3b,stroke:#34d399,color:#fff
    class health,profile,verification,checks api
    class gate,models,profile_pipeline,verification_pipeline,matcher engine
```

## Request flow

```mermaid
flowchart TD
    upload["Multipart image(s) or ZIP"] --> validate["Validate upload\nand decode images"]
    validate --> kind{"Document type"}
    kind --> passport["Passport profile\nMRZ-anchored page"]
    kind --> idcard["ID-card profile\nfront + back"]
    kind --> licence["Driving-licence profile"]
    passport --> localize["Localize and rectify"]
    idcard --> localize
    licence --> localize
    localize --> regions["Crop profile regions\nand field ROIs"]
    regions --> detect["Batched text detection"]
    detect --> filter["Keep lines inside\nvisible field regions"]
    filter --> recognize["Batched line recognition"]
    recognize --> parse["Assign fields and parse MRZ"]
    parse --> validate_result["Cross-field validation\nand confidence"]
    validate_result --> response["Ordered DocumentResult\nor isolated item error"]

    classDef step fill:#172554,stroke:#60a5fa,color:#fff
    classDef decision fill:#422006,stroke:#f59e0b,color:#fff
    classDef result fill:#064e3b,stroke:#34d399,color:#fff
    class upload,validate,localize,regions,detect,filter,recognize,parse,validate_result step
    class kind decision
    class response result
```

`app/main.py` is the composition root. `app/api/v1.py` owns HTTP transport,
safe in-memory ZIP expansion, asynchronous admission, and conversion to the
public schemas in `app/contracts.py` and `app/api/schemas.py`.

`app/inference/batch.py` is the shared coordinator. It groups compatible work
so DocAligner, MRZScanner, Paddle detection, and Paddle recognition receive
real tensors containing more than one sample when a request permits it. It
restores output ordering and isolates recoverable item failures. Separate HTTP
requests are admitted asynchronously but are not combined into one model call.

## Batching and failure isolation

```mermaid
flowchart LR
    request["One logical request\nwith multiple documents"] --> admit["Admission gate"]
    admit --> plan["Build stage jobs\nby compatible model + input"]
    plan --> tensor["Real tensor batch\nN > 1 when possible"]
    tensor --> run["Run shared model"]
    run --> split{"Item failed?"}
    split -->|"no"| restore["Restore original\ninput order"]
    split -->|"yes"| isolate["Record item error\nkeep siblings"]
    isolate --> restore
    restore --> response["Batch response\nresult or error per item"]

    classDef stage fill:#172554,stroke:#60a5fa,color:#fff
    classDef decision fill:#422006,stroke:#f59e0b,color:#fff
    classDef result fill:#064e3b,stroke:#34d399,color:#fff
    class request,admit,plan,tensor,run,restore stage
    class split decision
    class isolate,response result
```

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

The profile pipeline is the production extraction path. The verification OCR
path is intentionally narrower: it reuses only the text detector and Latin
recognizer, then returns raw lines. It does not localize documents or apply
profile geometry. The separate matcher accepts caller-supplied fields and may
derive additional MRZ evidence from valid MRZ-shaped lines already returned by
OCR; it does not run a specialized MRZ model.

## Document behavior

Production profile JSON lives in `config/documents/`. Passport and ID-card
pipelines in `app/documents/identity.py` own their regions, MRZ side/page, and
visible-to-MRZ reconciliation. Driving-licence parsing remains in
`app/documents/driving_license_fields.py`. Geometry, ROI assignment, confidence
aggregation, and artifact handling are shared.

OCR scores are returned with their source and
`calibrated_probability: false`; they are not represented as probabilities.

## Verification flow

```mermaid
flowchart TD
    input["Whole uploaded image(s)"] --> detect["Shared batched\ntext detection"]
    detect --> recognize["Shared batched Latin\nrecognition for every line"]
    recognize --> raw["Raw OCR lines\ngeometry + ocr_token score"]
    raw --> submit["Client submits OCR\nplus expected fields"]
    submit --> normalize["Normalize values\nand infer field kind"]
    normalize --> candidates["Build candidates\nline spans + adjacent lines"]
    candidates --> mrz["Optional valid MRZ evidence\nfrom supplied OCR lines"]
    mrz --> assign["Global one-to-one\nnon-overlapping assignment"]
    assign --> result["Per-field status\nscore source + evidence"]

    classDef model fill:#3f1d5b,stroke:#c084fc,color:#fff
    classDef data fill:#172554,stroke:#60a5fa,color:#fff
    classDef result fill:#064e3b,stroke:#34d399,color:#fff
    class detect,recognize model
    class input,raw,submit,normalize,candidates,mrz,assign data
    class result output
```

The verification transport owns batch admission and ID-card side pairing. Its
matcher owns normalization, type-aware scoring, adjacent-line candidates, and
constrained assignment. The verification OCR coordinator shares the
process-owned text detector and recognizer with the profile pipeline. No profile
geometry or field position participates in this flow.
