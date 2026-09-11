# Experiment index

Numbered reports in this directory record independent CPU-selection
experiments. Each report contains its own question, method, results, decision,
limits, reproduction command, and raw-data disposition. Reports do not depend
on or link to one another; this index is the only cross-report navigation.
Raw measurement artifacts live under the ignored `outputs/benchmarks/` tree.

| Report | Status | Decision relevance | Primary evidence |
|---|---|---|---|
| [01.OLD_PIPELINE](01.OLD_PIPELINE_14-August-2026_06-53.md) | historical | context for the recognizer change (D-018/D-021); no surviving default | `outputs/benchmarks/03.pipeline-stage-breakdown/{01.passport,02.id-card,03.driving-license}/01.full-production` |
| [02.MODEL_MATRIX](02.MODEL_MATRIX_19-August-2026_18-31.md) | current | selected `latin_PP-OCRv5_mobile_rec`; kept `PP-OCRv6_medium_det`, `fastvit_sa24`, generic-paddle MRZ (D-018, D-021) | `outputs/benchmarks/06.model-matrix-comparison/20260819T183142Z` |
| [03.FINAL_LATIN_PIPELINE](03.FINAL_LATIN_PIPELINE_21-August-2026_15-40.md) | historical | first tensor-level batching proof (D-006); no surviving default | `outputs/benchmarks/07.latin-pipeline-benchmark/20260821T154036Z` |
| [04.BATCH_SIZE_SWEEP](04.BATCH_SIZE_SWEEP_22-August-2026_04-03.md) | current | batch-size selection and correctness evidence (D-006, D-021); G6 reference artifact | `outputs/benchmarks/09.batch-size-sweep/20260822T140836Z` |
| [05.RECOGNIZER_AUDIT_AB](05.RECOGNIZER_AUDIT_AB_22-August-2026_19-21.md) | current | strongest causal recognizer evidence: 2.918× E2E, 97.2% of the gap in recognition (D-018, D-021) | `outputs/benchmarks/10.recognizer-a-b-comparison/20260822T142128Z` |
| [06.CPU_RUNTIME_AND_PREPROCESSING](06.CPU_RUNTIME_AND_PREPROCESSING_24-August-2026_11-16.md) | current | locked threads 4, packing, 960-side detector; MRZ `contrast_1.50` candidate (D-021); historical gate record deleted | five run families — see the report's §8 |

| [07.FULL_VERIFICATION_BASELINE](07.FULL_VERIFICATION_BASELINE_28-August-2026_05-26.md) | current | consolidated final best-vs-best Latin versus conservative matching comparison; Current wins correctness on all document types at a measured latency cost (D-021, D-026–D-030) | `outputs/benchmarks/22.latin-vs-current-matcher/20260902T112500Z` |
| [08.FULL_PIPELINE_VS_DIRECT_MATCHING](08.FULL_PIPELINE_VS_DIRECT_MATCHING_02-September-2026.md) | current | exact-vs-exact architecture trade-off: direct matching wins exact accuracy; full preprocessing wins latency; RSS comparison requires correction | `outputs/benchmarks/20.full-pipeline-vs-direct-matching/20260902T071921Z` |

## Figures

Figures are derived artifacts, regenerated from the cited raw outputs:

```bash
.venv/bin/python experiments/tools/make_figures.py
```

House style ([00.TEMPLATE.md](00.TEMPLATE.md)): an image carries data only —
marks, legend, axis names, value labels. Titles and descriptions live in the
reports as captions. Charts are chosen by the Pareto principle: every
decision-bearing shape gets a chart, nothing decorative. After the raw outputs
are deleted, the PNGs remain the archived evidence; the script can no longer
regenerate them.

## Raw-output disposition (as of 2026-09-02)

Per-report accounting lives in each report's §8. Summary:

- **Deletable now** (~1.0 GB): every duplicate rerun and superseded attempt
  listed in the reports' §8 tables — including the three `mrz_preprocessing_reconciliation`
  reruns (~289 MB), the two older `preprocessing` screening runs (~61 MB), and
  the four older `recognition_packing` runs (~28 MB) — plus the three
  retired figures `assets/{03_latin_batch_evidence,04_batch_size_pareto,05_recognizer_ab_pareto}.png`,
  which no report references anymore.
- **Conditional**: `outputs/benchmarks/11.cpu-thread-count-benchmark` — the historical gate record read its
  files directly. Delete only if the historical evidence is no longer needed.
- **Retained provenance, cited by no report** (~95 MB, the August reorg
  explicitly voted to preserve these):
  `outputs/benchmarks/{02.contrast-sweep,01.ocr-recognition-sweeps,04.six-optimization-comparison,05.split-ocr-batching-validation}`,
  `outputs/experiments/{alignment,mrz,roi}`. Deletable if you accept losing
  the raw backing of the early prototypes; the reports do not depend on them.

Reproducing the experiments: maintained tools live in `benchmarks/maintained/`;
the one-off drivers behind reports 02–06 are preserved in
`benchmarks/historical/` (`model_matrix_benchmark.py`, `batch_size_benchmark.py`,
`combined_batch_benchmark.py`, `cpu_thread_benchmark.py`,
`detector_resolution*.py`, `recognition_packing_benchmark.py`,
`recognizer_ab_benchmark.py`, `mrz_preprocessing_reconciliation.py`,
`preprocessing_benchmark.py`, `final_benchmark.py`).

Reports and generated figures are versioned. Raw measurements remain ignored
because they may contain local datasets and large process artifacts.
