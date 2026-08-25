# Configuration

Copy `.env.example` to `.env`. `app/config.py` validates values at startup.

| Group | Variables |
|---|---|
| Application | `VOIGHT_PORT`, `PRELOAD` |
| Runtime | `RUNTIME_TARGET`, `CPU_THREADS`, `GPU_ID` |
| Model cache | `MODEL_DIR` |
| Model selection | `TEXT_DETECTOR_*`, `TEXT_RECOGNIZER_*`, `DOCUMENT_LOCALIZER_BACKEND`, `DOCALIGNER_MODEL[_TYPE]`, `MRZ_LOCALIZER_BACKEND`, `MRZSCANNER_DETECTION_CFG`, `MRZ_RECOGNIZER_*` |
| Admission/batching | `REQUEST_QUEUE_LIMIT`, `LOCALIZATION_BATCH_SIZE`, `TEXT_DETECTION_BATCH_SIZE`, `TEXT_RECOGNITION_BATCH_SIZE`, `MRZ_RECOGNITION_BATCH_SIZE`, `TEXT_RECOGNITION_PROCESSES`, `TEXT_RECOGNITION_PACKING` |
| GPU experiments | `TEXT_RECOGNITION_ENABLE_HPI`, `TEXT_RECOGNITION_USE_TENSORRT`, `TEXT_RECOGNITION_PRECISION`, `TEXT_DETECTOR_PREPROCESSING`, `VISIBLE_RECOGNITION_PREPROCESSING`, `MRZ_PREPROCESSING` |
| Geometry/MRZ | `OCR_MAX_SIDE`, `OCR_CONTRAST`, `MRZ_POLYGON_PADDING_RATIO`, `DOCALIGNER_PADDING`, `DRIVING_LICENSE_CANONICAL_*`, `DRIVING_LICENSE_MIN_OVERLAP_RATIO` |
| Artifacts | `LOGGING`, `LOG_DIR` |
| Upload limits | `BATCH_MAX_FILES`, `BATCH_MAX_FILE_BYTES`, `BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES` |

`RUNTIME_TARGET` is authoritative. `OCR_DEVICE` is accepted only as a deprecated
fallback and should not be set in new environments. CPU rejects GPU-only HPI,
TensorRT, and FP16 options rather than silently ignoring them.

`REQUEST_QUEUE_LIMIT` includes the active request. Model batch sizes apply
inside one logical request; Python loops over one-image calls are not used as a
substitute for model batching. `TEXT_RECOGNITION_PROCESSES` may exceed one only
on CPU and should be changed only after measurement.

The selected CPU defaults are localization 4, detection 1, recognition 2,
`CPU_THREADS=4`, `OMP_NUM_THREADS=1`, and fixed-width recognition packing.
OpenCV threading is left at its runtime default. Detector and visible OCR use
original images. MRZ normalization is resize/grayscale-only; contrast 1.50 is
applied to each raw detected MRZ line immediately before fixed-width packing.

Profile paths also support `PASSPORT_PROFILE`, `ID_CARD_PROFILE`,
`DRIVING_LICENSE_DATA_CROP`, and `DRIVING_LICENSE_FIELD_ROIS`. Their project
defaults are normally correct and are omitted from `.env.example` to avoid
unnecessary local path overrides.
