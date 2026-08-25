# Gates: corrected CPU thread-count benchmark

Scope: complete dataset (9 passports/9 images, 4 logical ID cards/8 images, 7 driving licences/7 images), finalized L4-D1-R2 pipeline, thread counts 1/2/4/6/8/12/16, three measured repeats after one warm-up, and fresh-container lifecycle checks.

- [x] G1: Dataset manifest and machine CPU counts are recorded
  CHECK: python -c "import json; x=json.load(open('outputs/benchmarks/cpu_thread_benchmark/system.json')); assert x['dataset']['expected_logical_documents']=={'passport':9,'id-card':4,'driving-license':7}; assert x['dataset']['expected_physical_images']=={'passport':9,'id-card':8,'driving-license':7}; assert x['cpu']['physical_cpu_count'] and x['cpu']['logical_cpu_count']"
  EVIDENCE: system.json

- [x] G2: Every startup records effective settings and runtime thread behavior
  CHECK: test "$(find outputs/benchmarks/cpu_thread_benchmark/runs -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 7 && test "$(find outputs/benchmarks/cpu_thread_benchmark/runs -name startup.json | wc -l)" -eq 7 && rg -l 'onnx_session_options|opencv|paddle_device|cpu_threads' outputs/benchmarks/cpu_thread_benchmark/runs/*/startup.json | wc -l
  EVIDENCE: runs/*/startup.json

- [x] G3: All 63 measured requests contain the complete expected workload
  CHECK: python -c "import json; r=json.load(open('outputs/benchmarks/cpu_thread_benchmark/raw_results.json')); assert len(r)==63; assert all(x['request_verification']['response_counts_match'] and x['request_verification']['submitted_logical_documents']=={'passport':9,'id-card':4,'driving-license':7}[x['document_type']] and x['request_verification']['submitted_physical_images']=={'passport':9,'id-card':8,'driving-license':7}[x['document_type']] for x in r)"
  EVIDENCE: raw_results.json and runs/<threads>/*_repeat_*.json

- [x] G4: Correct throughput, timing, RSS, CPU, tensor batches, correctness, MRZ, failures, and baseline diffs are present
  CHECK: test "$(wc -l < outputs/benchmarks/cpu_thread_benchmark/comparison.csv)" -eq 22 && rg -n 'docs_per_second_median|localization_seconds_median|actual_tensor_batches|character_accuracy|mrz_exact_items|output_differences_vs_4_threads|semantic_output_differences_vs_4_threads' outputs/benchmarks/cpu_thread_benchmark/comparison.csv
  EVIDENCE: comparison.csv and raw_results.json

- [x] G5: Every server was stopped, removed, and process termination verified
  CHECK: python -c "import json,glob; x=[json.load(open(p)) for p in glob.glob('outputs/benchmarks/cpu_thread_benchmark/runs/*/lifecycle.json')]; assert len(x)==7 and all(y['process_termination_verified'] and y['container_removed'] for y in x)"
  EVIDENCE: runs/*/lifecycle.json

- [x] G6: Four-thread sanity check against the existing full-dataset L4-D1-R2 artifact is within 20% for every document type
  CHECK: python -c "import json; x=json.load(open('outputs/benchmarks/cpu_thread_benchmark/sanity_check.json')); assert x['assessment']=='consistent_with_reference'"
  EVIDENCE: sanity_check.json; reference: outputs/benchmarks/batch_size/20260822T140836Z/summary.json
