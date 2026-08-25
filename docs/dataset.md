# Datasets and annotation

Identity documents contain sensitive personal information. Use only data you
are authorized to possess and process. The repository's committed
`annotation_input/` and `annotations/` material is authorized
anonymized or openly sourced fixture data. New local data belongs in ignored
`dataset/` and must not be committed by default.

## Ground-truth dataset

The resumable document-level tool and exact directory layout are documented in
`scripts/dataset/README.md`:

```bash
make dataset-annotate
make dataset-summary
make dataset-validate
make dataset-export
```

It never rewrites source images, records SHA-256 values, saves each accepted
field atomically, keeps visible fields and MRZ truth independent, and exports
secondary JSONL under ignored `dataset/exports/`.

## Profile geometry

The guided OpenCV annotator maintains production crop/ROI geometry:

```bash
uv run --no-sync python scripts/dataset/annotate_profiles.py annotation_input annotations
uv run --no-sync python scripts/dataset/annotate_profiles.py annotation_input annotations --check
uv run --no-sync python scripts/dataset/promote_profiles.py
uv run --no-sync python scripts/dataset/promote_profiles.py --check
```

`annotations/annotation_state.json` is the source of truth for the supplied
profiles. Promotion writes only reusable geometry/schema to `config/`; expected
personal values remain outside production configuration. Passport page corners
are stored relative to the detected MRZ, and ID-card front/back are one logical
document with separate side ownership.
