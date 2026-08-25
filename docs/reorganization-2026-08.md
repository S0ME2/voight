# Repository reorganization — August 2026

This plan reorganizes the repository so that maintained tooling, historical
benchmark tooling, experiment evidence, and archived exploration code are
clearly separated, without changing application behavior. Baseline before any
change: `make check` exit 0 and `make test` with 112 tests OK.

## A. Current structural problems

Verified against the tree on 2026-08-24:

1. `scripts/benchmarking/` mixes 23 scripts (9,660 LOC) of three different
   kinds: the shared measurement foundation (`pipeline_breakdown.py`,
   unit-tested), Makefile/docs-wired tools
   (`text_recognition_batch_benchmark.py`, `preprocessing_benchmark.py`),
   reusable server-lifecycle drivers (`model_matrix_benchmark.py`,
   `batch_size_benchmark.py`), and sixteen one-off drivers for completed
   experiments.
2. The maintained end-to-end API benchmark lives in a top-level `complexity/`
   directory whose name describes nothing (`make benchmark-cpu`,
   `docs/benchmarking.md`, README project map all point there). Four other
   benchmark scripts also default their results into `complexity/results/`.
3. `scripts/experiments/` holds early exploration prototypes (DocAligner,
   UVDoc, FastMRZ, MRZScanner, OCR sweeps, ROI experiments) that were superseded
   by `app/documents/*`. One file (`mrz/annotate_passport_mrz.py`) is still
   load-bearing: `tests/test_passport_mrz_annotation.py` imports
   `relative_to_mrz` from it.
4. `GATES.md` sits at the repository root, untracked and unreferenced. It is
   the completed acceptance-gate checklist (G1–G6) of the CPU thread-count
   benchmark that locked `CPU_THREADS=4` (experiment 06, track 1).
5. Leftover empty directories misrepresent structure: `app/tools/`,
   `scripts/pipelines/`, `scripts/tools/{alignment,roi}/` contain no files and
   are not tracked by git.
6. The local `experiments/` report area (git-ignored) has six numbered reports
   but no index stating which are current, superseded, or historical, and one
   acknowledged typo in an artifact path inside report 06
   (`202604T111635Z` should read `20260824T111635Z`).
7. `tools/check_runtime.py`, `scripts/dataset/augment_dataset.py`,
   `requirements/experiments.txt` are each referenced from nowhere operational;
   they are preserved and documented rather than moved or deleted (see D/E).

## B. Proposed target structure

```text
app/                    unchanged production code (no changes at all)
config/                 unchanged
tests/                  unchanged layout; imports updated to new tool paths
docs/                   operating documentation + this plan + docs/gates.md
scripts/
    dataset/            annotation/dataset tooling (+ annotate_passport_mrz.py)
    models/             model cache preparation (production + benchmark)
    validation/         local validation evidence tool
benchmarks/
    maintained/         general-purpose benchmarks for future development
    historical/         one-off drivers reproducing completed experiments
archive/
    scripts-experiments/  early exploration prototypes (kept, not deleted)
    assets-legacy/        unreferenced/duplicate tracked assets (kept, not deleted)
experiments/            git-ignored local reports + new README.md index
.codex/                 governance (ignored); STATUS.md gains a dated entry
```

`app/`, `config/`, `annotations/`, `annotation_input/`, `dataset/`,
and `models/` are not touched. `tools/check_runtime.py` stays at `tools/`
(uncertain classification; see D). The `assets/` directory is eliminated —
its tracked files archived to `archive/assets-legacy/`, untracked duplicates
listed as deletion candidates. No `benchmarks/common/` module is extracted:
the shared server lifecycle already lives in maintained scripts that everyone
imports in a healthy direction (historical → maintained); extracting helpers
would be a benchmark-framework rewrite, which is deferred (see F).

## C. Old → new mapping

Tracked files move with `git mv`; untracked files move with plain `mv`
(no git history exists for them). Classification: M = maintained,
H = historical/experiment reproduction, A = archived exploration.

| Old path | New path | Cl. | Reason |
|---|---|---|---|
| `complexity/benchmark_batch_complexity.py` | `benchmarks/maintained/benchmark_batch_complexity.py` | M | Maintained end-to-end benchmark (`make benchmark-cpu`); `complexity/` name describes nothing |
| `scripts/benchmarking/pipeline_breakdown.py` | `benchmarks/maintained/pipeline_breakdown.py` | M | Shared measurement foundation; unit-tested; actively developed |
| `scripts/benchmarking/text_recognition_batch_benchmark.py` | `benchmarks/maintained/text_recognition_batch_benchmark.py` | M | Makefile `benchmark-recognition` target; supported recognition benchmark |
| `scripts/benchmarking/split_ocr_validation.py` | `benchmarks/maintained/split_ocr_validation.py` | M | Unit-tested batching-equivalence validation harness |
| `scripts/benchmarking/v1_batch_benchmark.py` | `benchmarks/maintained/v1_batch_benchmark.py` | M | Generic load test for `/v1` batch routes; needed again during V100 task 010 |
| `scripts/benchmarking/model_matrix_benchmark.py` | `benchmarks/maintained/model_matrix_benchmark.py` | M | Canonical server-lifecycle/helper library for seven other benchmarks; rerunnable model comparison |
| `scripts/benchmarking/batch_size_benchmark.py` | `benchmarks/maintained/batch_size_benchmark.py` | M | Stage batch-size sweep; will be reused for GPU micro-batch tuning (D-008) |
| `scripts/benchmarking/preprocessing_benchmark.py` | `benchmarks/maintained/preprocessing_benchmark.py` | M | Documented preprocessing benchmark with resume modes (`docs/benchmarking.md`) |
| `scripts/benchmarking/benchmark.py` | `benchmarks/historical/benchmark.py` | H | Historical model/preprocessing selection sweep (stated in docs) |
| `scripts/benchmarking/contrast_benchmark.py` | `benchmarks/historical/contrast_benchmark.py` | H | Historical contrast sweep (stated in docs) |
| `scripts/benchmarking/final_benchmark.py` | `benchmarks/historical/final_benchmark.py` | H | Historical final-config picker over `benchmark.py` |
| `scripts/benchmarking/optimization_six.py` | `benchmarks/historical/optimization_six.py` | H | Driver of the closed "six optimizations" run; frozen timestamp |
| `scripts/benchmarking/analyze_optimization_six.py` | `benchmarks/historical/analyze_optimization_six.py` | H | Post-hoc analysis of that same frozen run |
| `scripts/benchmarking/mrz_recognition_benchmark.py` | `benchmarks/historical/mrz_recognition_benchmark.py` | H | Two-image generic-vs-specialized MRZ A/B; question settled (D-018) |
| `scripts/benchmarking/mrz_paddle_batch_benchmark.py` | `benchmarks/historical/mrz_paddle_batch_benchmark.py` | H | MRZ batch sweep; superseded by corrected batch_size methodology |
| `scripts/benchmarking/recognizer_comparison_benchmark.py` | `benchmarks/historical/recognizer_comparison_benchmark.py` | H | Recognizer×packing screen superseded by strict A/B (experiment 05) |
| `scripts/benchmarking/combined_batch_benchmark.py` | `benchmarks/historical/combined_batch_benchmark.py` | H | Four hand-picked combined tuples vs one fixed baseline run; closed |
| `scripts/benchmarking/cpu_thread_benchmark.py` | `benchmarks/historical/cpu_thread_benchmark.py` | H | Reproduces the completed thread-count experiment gated by GATES G1–G6 |
| `scripts/benchmarking/detector_resolution_benchmark.py` | `benchmarks/historical/detector_resolution_benchmark.py` | H | Detector downscale sweep; concluded "keep 960" (experiment 06 track 3) |
| `scripts/benchmarking/detector_resolution_workload_benchmark.py` | `benchmarks/historical/detector_resolution_workload_benchmark.py` | H | Follow-up tensor-pixel-count search of the same question |
| `scripts/benchmarking/finalize_detector_resolution_workload.py` | `benchmarks/historical/finalize_detector_resolution_workload.py` | H | Report-recovery twin for one interrupted workload run |
| `scripts/benchmarking/recognition_packing_benchmark.py` | `benchmarks/historical/recognition_packing_benchmark.py` | H | Packing strategy comparison; conclusion locked as `fixed-width` (D-021) |
| `scripts/benchmarking/recognizer_ab_benchmark.py` | `benchmarks/historical/recognizer_ab_benchmark.py` | H | Strict recognizer A/B; conclusion locked (D-021, experiment 05) |
| `scripts/benchmarking/mrz_preprocessing_reconciliation.py` | `benchmarks/historical/mrz_preprocessing_reconciliation.py` | H | Reconciled MRZ-line preprocessing; conclusion locked (contrast 1.50, D-021) |
| `scripts/experiments/mrz/annotate_passport_mrz.py` | `scripts/dataset/annotate_passport_mrz.py` | M→ops | Still imported by a test; diagnostic companion of the annotation workflow (D-013 anchor geometry) |
| `scripts/experiments/alignment/test_docaligner.py` | `archive/scripts-experiments/alignment/test_docaligner.py` | A | Superseded exploration prototype |
| `scripts/experiments/alignment/test_uvdoc.py` | `archive/scripts-experiments/alignment/test_uvdoc.py` | A | Superseded exploration prototype |
| `scripts/experiments/mrz/test_fastmrz.py` | `archive/scripts-experiments/mrz/test_fastmrz.py` | A | FastMRZ exploration; sole consumer of `requirements/experiments.txt` |
| `scripts/experiments/mrz/test_mrzscanner.py` | `archive/scripts-experiments/mrz/test_mrzscanner.py` | A | MRZScanner exploration prototype |
| `scripts/experiments/ocr/first_run.py` | `archive/scripts-experiments/ocr/first_run.py` | A | First OCR sweep over `dataset/source/` |
| `scripts/experiments/ocr/best_run.py` | `archive/scripts-experiments/ocr/best_run.py` | A | Tuned OCR sweep; superseded by profile pipeline |
| `scripts/experiments/roi/test_data_crop.py` | `archive/scripts-experiments/roi/test_data_crop.py` | A | ROI exploration prototype |
| `scripts/experiments/roi/test_fields_extraction.py` | `archive/scripts-experiments/roi/test_fields_extraction.py` | A | Prototype of today's field assignment logic |
| `scripts/experiments/roi/test_rois.py` | `archive/scripts-experiments/roi/test_rois.py` | A | Earlier canonical-ROI variant |
| `GATES.md` (untracked) | `docs/gates.md` | doc | Completed acceptance gates belong with documentation; untracked, so plain `mv` |
| `assets/templates/` | `archive/assets-legacy/templates/` | A | Unreferenced license templates (tracked) |
| `assets/samples/id_card/idcard.jpeg` | `archive/assets-legacy/id_card/idcard.jpeg` | A | Unreferenced demo image (tracked) |
| `assets/samples/id_card/idcard2.jpeg` | `archive/assets-legacy/id_card/idcard2.jpeg` | A | Unreferenced demo image (tracked) |
| `assets/samples/passport/uzpassport_crop.png` | `archive/assets-legacy/passport/uzpassport_crop.png` | A | Unreferenced derived crop (tracked) |
| `assets/samples/passport/passport.png` | `archive/assets-legacy/passport/passport.png` | A | Duplicate of `annotation_input/passports/passport.png` (tracked) |

Reference updates shipped in the same working-tree change:

- Moved benchmark scripts: cross-script imports change from
  `scripts.benchmarking.X` to `benchmarks.maintained.X` /
  `benchmarks.historical.X`; default result paths `complexity/results/…` in
  four scripts become `benchmarks/results/…`;
  `benchmark_batch_complexity.py` ROOT depth `parents[1]` → `parents[2]`;
  `annotate_passport_mrz.py` sys.path depth `parents[3]` → `parents[2]`.
- Tests: `test_pipeline_breakdown.py`, `test_split_ocr_validation.py`,
  `test_passport_mrz_annotation.py` import paths.
- `Makefile`: `benchmark-cpu`, `benchmark-recognition` script paths; `check`
  compileall adds `benchmarks archive`; `clean` find list likewise.
- `.gitignore`: `complexity/results/` → `/benchmarks/results/` (no
  `complexity/results/` data exists on disk).
- `.dockerignore`: `complexity` → `benchmarks`, `archive` (neither is needed
  in any image build context).
- `README.md`: project map rows for `complexity/`, `scripts/`, plus
  `benchmarks/` and `archive/`.
- `docs/benchmarking.md`: paths, results location, historical note now points
  at `benchmarks/historical/`, link to `docs/gates.md`, pointer to the local
  experiment index.
- `docs/development.md`: "Historical experiments" paragraph points to
  `archive/scripts-experiments/`.
- `experiments/06.CPU_RUNTIME_AND_PREPROCESSING_24-August-2026_11-16.md`:
  artifact-path typo fix only (`202604T111635Z` → `20260824T111635Z`).
- New: `experiments/README.md` index with per-report status, decision
  relevance, and evidence links.
- `.codex/STATUS.md`: dated reorganization entry. `.codex/tasks/TASK-009`
  mentions `complexity/` historically and is left as written (task cards are
  records, not living docs).

## D. Files deliberately left untouched

- `dataset/` — sensitive local data; never opened beyond existence checks.
- `models/` — local model caches; not part of layout cleanup.
- `outputs/`, `logs/` — ignored generated data; referenced artifacts must keep
  resolving (all paths cited by experiment reports and `docs/gates.md` were
  verified to exist).
- `annotation_input/`, `annotations/` — canonical source fixtures and annotator
  output; all paths reference these, so they stay.
- `tools/check_runtime.py` — unwired runtime/readiness debug printer; plausibly
  useful during V100 task 010, so preserved in place rather than archived.
  Uncertainty documented here instead of resolved by moving it.
- `scripts/dataset/augment_dataset.py` — currently unreferenced, but synthetic
  augmentation has a named future role in D-003; kept next to the dataset
  tooling it belongs to.
- `requirements/experiments.txt` — orphaned from every `-r` chain but mirrored
  by the uv `experiments` group and consumed by the archived FastMRZ script;
  kept so that archived code remains runnable.
- `experiments/*.md` report bodies — conclusions untouched except the one path
  typo; statuses recorded in the new index instead of rewriting reports.
- `app/` — zero changes; nothing under `app/` imports any moved file.

## E. Deletion / archive candidates

Archived (moved, preserved): the nine `scripts/experiments/` prototypes listed
in C, plus tracked asset files moved to `archive/assets-legacy/`. Nothing
tracked was deleted.

Deletion candidates (NOT executed; listed for manual review):

1. Empty untracked directories `app/tools/`, `scripts/pipelines/`,
   `scripts/tools/{alignment,roi}/`, plus the now-sourceless cache shells
   `scripts/benchmarking/__pycache__/` and `scripts/experiments/*/__pycache__/`
   — they contain no tracked or generated data of any value; git has never
   tracked them; `scripts/pipelines/` is already asserted legacy by the layout
   test's removed-file check. Recommended for manual removal with the command
   in the reorganization report (directory deletion was not executed during
   this cleanup).
2. Stray `__pycache__/` directories elsewhere — regenerable bytecode caches.
   Optional; `make clean` already removes them.
3. `outputs/benchmarks/{contrast,ocr,optimization_six,detector_resolution,split_ocr_validation}/`
   and `outputs/experiments/` — earlier/superseded raw runs not cited by the
   six numbered reports. NOT recommended for deletion: they are experimental
   provenance even when unreferenced, and disk cost is modest (~57 MB of the
   780 MB benchmark tree). Preserve.
4. Untracked duplicate/orphan files remaining in `assets/`:
   - `assets/samples/passport/uzpassport.png` — byte-identical duplicate of
     `annotation_input/passports/passport.png`
   - `assets/samples/driving_license/test_license_data_crop.jpg` — unreferenced
   - `assets/samples/driving_license/test_license.jpg` — unreferenced (only
     `test_license_canonical.jpg` is used)
   - The now-empty directory shells `assets/samples/driving_license/`,
     `assets/samples/id_card/`, `assets/samples/passport/`, `assets/samples/`,
     `assets/`
   Recommended for deletion.
5. `uv.lock` entry churn, `.gitignore` legacy lines (`DELETE_ME/`) — cosmetic
   only; left alone.

Manual command (I did **not** execute it):

```bash
rm -f assets/samples/passport/uzpassport.png assets/samples/driving_license/test_license_data_crop.jpg assets/samples/driving_license/test_license.jpg && \
rmdir assets/samples/driving_license assets/samples/id_card assets/samples/passport assets/samples assets app/tools scripts/pipelines scripts/tools/alignment scripts/tools/roi scripts/tools
```
   provenance even when unreferenced, and disk cost is modest (~57 MB of the
   780 MB benchmark tree). Preserve.
4. Untracked duplicate/orphan files remaining in `assets/`:
   - `assets/samples/passport/uzpassport.png` — byte-identical duplicate of
     `annotation_input/passports/passport.png`
   - `assets/samples/driving_license/test_license_data_crop.jpg` — unreferenced
   - `assets/samples/driving_license/test_license.jpg` — unreferenced (only
     `test_license_canonical.jpg` is used)
   - The now-empty directory shells `assets/samples/driving_license/`,
     `assets/samples/id_card/`, `assets/samples/passport/`, `assets/samples/`,
     `assets/`
   **User deleted these manually after the plan was written.** Verified gone;
   `.dockerignore` and `README.md` project map updated to remove the `assets/`
   entry; `docs/dataset.md` updated.
5. `uv.lock` entry churn, `.gitignore` legacy lines (`DELETE_ME/`) — cosmetic
   only; left alone.

## F. Deferred work (out of scope here)

- Benchmark helper duplication: Levenshtein implemented four-plus times,
  `BASE_ENV` blocks repeated ~7× with drifting values, CSV/JSON writers and
  environment capture duplicated, two incompatible server lifecycles
  (subprocess `Server` vs Docker-container driver). Extracting a
  `benchmarks/common/` module would touch every measurement script and is
  deliberately deferred; the healthy import direction (historical → maintained)
  keeps this safe to do later.
- `tests/test_pipeline_breakdown.py` imports `fixed_rows` from the historical
  `optimization_six.py`; if that historical driver is ever removed, drop the
  corresponding test first.
- `tools/check_runtime.py` wiring (or archival) once task 010 shows whether it
  is wanted.
- `docs/dataset.md` documents `annotate_profiles.py` while the Makefile drives
  `annotate.py`; both are correct entry points for different flows, but a
  single clarified flow would help.

## G. Validation performed after execution

- `make check` (compileall over `app scripts tests benchmarks archive`, lock
  check, whitespace check)
- `make test` (full CPU suite; expected green, same count as baseline)
- explicit `python -m compileall -q app scripts tests benchmarks archive`
- per-move stale-reference audit driven by table C (tracked sources, docs,
  Makefile, Dockerfile, shell, tests, `.codex/`), allowing only intentional
  historical mentions
- `git status`/`git diff` review confirming no deletions, no app/ edits, no
  commits.
