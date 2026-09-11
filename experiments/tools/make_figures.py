#!/usr/bin/env python3
"""Regenerate the experiment-report figures in experiments/assets/ from raw data.

Every figure is rebuilt from the cited measurement files under outputs/, so the
PNGs are derived artifacts and the reports' numbers stay traceable. Run from the
repository root:

    .venv/bin/python experiments/tools/make_figures.py

House style: an image carries data only — marks, legend, axis names, and direct
value labels. Titles, experiment names, descriptions, and caveats live in the
reports as figure captions, never inside the PNG. Panel titles are allowed only
where several panels must be told apart.

Required inputs, per figure:

  01   outputs/benchmarks/03.pipeline-stage-breakdown/{01.passport,02.id-card,03.driving-license}/
         01.full-production/{stage_breakdown.csv,summary.json}
  02a  outputs/benchmarks/06.model-matrix-comparison/20260819T183142Z/aggregate.csv
  02b  outputs/benchmarks/06.model-matrix-comparison/20260819T183142Z/aggregate.csv
  03   outputs/benchmarks/07.latin-pipeline-benchmark/20260821T154036Z/
         {01.driving-license,02.id-card,03.passport}/{stage_breakdown.csv,raw_measurements.csv}
  04a  outputs/benchmarks/09.batch-size-sweep/20260822T140836Z/summary.json
  04b  outputs/benchmarks/09.batch-size-sweep/20260822T140836Z/summary.json
  05a  outputs/benchmarks/10.recognizer-a-b-comparison/20260822T142128Z/summary.json
  05b  outputs/benchmarks/10.recognizer-a-b-comparison/20260822T142128Z/summary.json
       outputs/benchmarks/08.recognizer-final-confirmation/20260822T181139Z/aggregate.json
  06a  outputs/benchmarks/11.cpu-thread-count-benchmark/comparison.csv
  06b  outputs/benchmarks/12.recognition-packing-comparison/20260824T075726Z/{aggregate.csv,output_differences.json}
  06c  outputs/benchmarks/13.detector-resolution-workload/20260824T090238Z/broad_sweep.csv
  06d  outputs/benchmarks/14.mrz-preprocessing-reconciliation/20260824T111635Z/mrz_aggregate.json
  07   outputs/benchmarks/22.latin-vs-current-matcher/20260902T112500Z/{document_type_accuracy.csv,
       failure_transitions.csv,raw_runs.csv,stage_timings.csv}
  08   outputs/benchmarks/20.full-pipeline-vs-direct-matching/20260902T071921Z/{overall_accuracy.csv,
       document_type_accuracy.csv,performance.csv,raw_runs.csv,detection_workload.csv,
       recognition_workload.csv}; outputs/benchmarks/21.full-pipeline-vs-direct-matching-audit/
       20260902T080411Z/timing_accounting.csv

Style: restrained light-surface palette with fixed semantic colors. Stage colors
and document-type colors are fixed across all figures; color follows the entity,
never its rank. The generator is deliberately dependency-light: matplotlib and
numpy are the only plotting dependencies.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(50_000_000)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "experiments" / "assets"

# --- palette (validated; see experiments/00.TEMPLATE.md) --------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

STAGE_COLORS = {
    "Localization": "#4a3aa7",   # violet
    "Detection": "#eb6834",      # orange
    "Recognition": "#2a78d6",    # blue
    "Other": "#898781",          # neutral fold, not a series hue
}
DOC_COLORS = {
    "passport": "#2a78d6",
    "id_card": "#eb6834",
    "driving_license": "#1baf7a",
}
DOC_LABELS = {
    "passport": "Passport",
    "id_card": "ID card",
    "driving_license": "Driving licence",
}
ARCH_STAGE_COLORS = {
    "Localization": "#4a3aa7",
    "MRZ": "#eda100",
    "Canonicalization + crop": "#8b5cf6",
    "Detection": "#eb6834",
    "Recognition": "#2a78d6",
    "Matching": "#1baf7a",
    "Other": "#898781",
}
FOUR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CANDIDATE = "#2a78d6"
INCUMBENT = "#eb6834"
SELECTED = "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def new_fig(w=7.2, h=4.2):
    return plt.subplots(figsize=(w, h), dpi=180, layout="constrained")


def style(ax, ygrid=True):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(0.9)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)


def legend(ax, **kw):
    return ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kw)


def highlight(ax, x, y, color, *, size=105):
    """Use a quiet ring for a reference/selected point; avoid novelty markers."""
    ax.scatter([x], [y], s=size, facecolor=SURFACE, edgecolor=color,
               linewidth=2.4, zorder=3)


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


KEEP_STAGES = ("localization", "detection", "recognition")
STAGE_LABEL = {"localization": "Localization", "detection": "Detection", "recognition": "Recognition"}
PIPELINE_DOC_DIR = {"passport": "01.passport", "id_card": "02.id-card", "driving_license": "03.driving-license"}
FINAL_LATIN_DOC_DIR = {"passport": "03.passport", "id_card": "02.id-card", "driving_license": "01.driving-license"}


# --- 01 / 03: stage shares ---------------------------------------------------

def wall_totals_passport_breakdown():
    sizes = {"passport": 9, "id_card": 4, "driving_license": 7}
    out = {}
    for doc, n in sizes.items():
        summary = json.load(open(
            ROOT / f"outputs/benchmarks/03.pipeline-stage-breakdown/{PIPELINE_DOC_DIR[doc]}/01.full-production/summary.json"))
        stats = summary["stats"][f"{doc}_full_production"]["by_logical_count"][str(n)]
        out[doc] = (n, stats["total_seconds"]["median"])
    return out


def wall_totals_final_latin():
    sizes = {"passport": 9, "id_card": 4, "driving_license": 7}
    out = {}
    for doc, n in sizes.items():
        rows = [r for r in read_csv(
            ROOT / f"outputs/benchmarks/07.latin-pipeline-benchmark/20260821T154036Z/{FINAL_LATIN_DOC_DIR[doc]}/raw_measurements.csv")
            if r["logical_count"] == str(n)]
        out[doc] = (n, median(float(r["server_seconds"]) for r in rows))
    return out


def stage_shares(csv_paths, totals):
    out = {}
    for doc, path in csv_paths.items():
        rows = [r for r in read_csv(path) if r["logical_count"] == str(totals[doc][0])]
        per_stage = defaultdict(list)
        for r in rows:
            per_stage[r["stage"].strip()].append(float(r["seconds"]) / totals[doc][1])
        med = {s: median(v) for s, v in per_stage.items()}
        kept = [(STAGE_LABEL[s], med[s]) for s in KEEP_STAGES if s in med]
        kept.append(("Other", sum(v for s, v in med.items() if s not in KEEP_STAGES)))
        out[doc] = kept
    return out


def draw_stage_share(name, csv_paths, totals):
    shares = stage_shares(csv_paths, totals)
    docs = list(shares)
    fig, ax = new_fig(6.6, 4.6)
    x = np.arange(len(docs))
    bottom = np.zeros(len(docs))
    for stage in ["Localization", "Detection", "Other", "Recognition"]:
        vals = np.array([next(v for l, v in shares[d] if l == stage) * 100 for d in docs])
        ax.bar(x, vals, 0.52, bottom=bottom, color=STAGE_COLORS[stage],
               label=stage, edgecolor=SURFACE, linewidth=2)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if stage == "Recognition" or v >= 9:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        color="#ffffff", fontsize=9, fontweight="bold")
        bottom += vals
    for xi, d in enumerate(docs):
        ax.text(xi, 101.5, f"{totals[d][1]:.2f} s", ha="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_xticks(x, [DOC_LABELS[d] for d in docs])
    ax.set_ylabel("Share of median request time (%)")
    ax.set_ylim(0, 100)
    style(ax)
    legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    save(fig, name)


def fig01():
    base = ROOT / "outputs/benchmarks/03.pipeline-stage-breakdown"
    paths = {d: base / PIPELINE_DOC_DIR[d] / "01.full-production" / "stage_breakdown.csv"
             for d in ("passport", "id_card", "driving_license")}
    draw_stage_share("01_old_pipeline_stage_breakdown.png", paths,
                     wall_totals_passport_breakdown())


def fig03():
    base = ROOT / "outputs/benchmarks/07.latin-pipeline-benchmark/20260821T154036Z"
    paths = {d: base / FINAL_LATIN_DOC_DIR[d] / "stage_breakdown.csv"
             for d in ("passport", "id_card", "driving_license")}
    draw_stage_share("03_latin_stage_breakdown_share.png", paths,
                     wall_totals_final_latin())


# --- 02: model matrix --------------------------------------------------------

def short_rec(model):
    m = model.replace("PP-OCRv", "v").replace("_rec", "").replace("_mobile", " mobile")
    return m.replace("latin_", "latin ").replace("en_", "en ").replace("cyrillic_", "cyr ").replace("_", " ")


FIG02_OFFSETS = {
    "v6 medium": (10, 10), "v6 small": (8, -2), "v6 tiny": (8, -4),
    "latin v5 mobile": (8, 6), "en v5 mobile": (8, 6), "eslav v5 mobile": (-118, -8),
    "cyr v5 mobile": (8, -10), "v5 server": (8, 0), "v4 server": (8, -4),
    "en v4 mobile": (8, 6), "v5 mobile": (8, 8), "v4 mobile": (8, -12),
}


def fig02a(rec_rows, ref):
    fig, ax = new_fig(7.4, 4.8)
    front = sorted(
        [c for c in rec_rows if not any(o[1] >= c[1] and o[2] >= c[2] and o != c
                                        for o in rec_rows)],
        key=lambda c: c[1])
    if len(front) > 1:
        ax.plot([c[1] for c in front], [c[2] for c in front], ls="--", lw=1.4,
                color=MUTED, zorder=1)
    ax.scatter([c[1] for c in rec_rows], [c[2] for c in rec_rows], s=46,
               color=CANDIDATE, zorder=2)
    selected = next((c for c in rec_rows if c[0] == "latin v5 mobile"), None)
    if selected:
        highlight(ax, selected[1], selected[2], SELECTED)
    for name, x, y in rec_rows:
        dx, dy = FIG02_OFFSETS.get(name, (7, 4))
        label = f"{name} · selected" if name == "latin v5 mobile" else name
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=INK_2)
    if ref:
        ax.scatter([ref[0]], [ref[1]], s=105, color=INCUMBENT, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate("baseline", ref, textcoords="offset points",
                    xytext=(15, -17), fontsize=8, color=INK_2, ha="left")
    ax.set_xlabel("Throughput (logical documents/second) — higher is faster")
    ax.set_ylabel("Exact visible fields (%)")
    ax.set_xlim(max(0.25, min(c[1] for c in rec_rows) - 0.08),
                max(c[1] for c in rec_rows) + 0.08)
    ax.set_ylim(max(0, min(c[2] for c in rec_rows) - 3),
                min(100, max(c[2] for c in rec_rows) + 3))
    style(ax)
    save(fig, "02_model_matrix_recognizer_tradeoff.png")


DET_OFFSETS = {
    "PP-OCRv6_medium_det": (10, -4), "PP-OCRv6_small_det": (10, 0),
    "PP-OCRv6_tiny_det": (10, 0), "PP-OCRv5_server_det": (-10, 6),
    "PP-OCRv5_mobile_det": (10, -4), "PP-OCRv4_server_det": (-10, 6),
    "PP-OCRv4_mobile_det": (-10, -12),
}


def fig02b(det_rows, ref):
    fig, ax = new_fig(7.0, 4.6)
    ax.scatter([c[1] for c in det_rows], [c[2] for c in det_rows], s=46,
               color=CANDIDATE, zorder=2)
    for name, x, y in det_rows:
        dx, dy = DET_OFFSETS.get(name, (8, 4))
        ha = "left" if dx > 0 else "right"
        ax.annotate(name.replace("_det", "").replace("_", " "), (x, y),
                    textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, color=INK_2, ha=ha)
    if ref:
        ax.scatter([ref[0]], [ref[1]], s=105, color=INCUMBENT, zorder=3,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate("baseline", ref, textcoords="offset points", xytext=(15, -17),
                    fontsize=8, color=INK_2, ha="left")
    ax.set_xlabel("Median full-corpus latency (seconds) — lower is better")
    ax.set_ylabel("Exact visible fields (%)")
    x_values = [c[1] for c in det_rows]
    y_values = [c[2] for c in det_rows]
    ax.set_xlim(min(x_values) - 2, max(x_values) + 2)
    ax.set_ylim(max(0, min(y_values) - 2), min(100, max(y_values) + 2))
    style(ax)
    save(fig, "02_model_matrix_detector_tradeoff.png")


def fig02():
    rows = read_csv(ROOT / "outputs/benchmarks/06.model-matrix-comparison/20260819T183142Z/aggregate.csv")
    rec, det, ref_rec, ref_det = [], [], None, None
    for r in rows:
        if r["status"] != "ok":
            continue
        try:
            y = 100.0 * int(r["visible_field_exact_count"]) / int(r["visible_field_total"])
        except (ValueError, ZeroDivisionError):
            continue
        if r["category"] == "recognizer":
            try:
                rec.append((short_rec(r["model"]),
                            float(r["throughput_docs_per_second"]), y))
            except ValueError:
                pass
        elif r["category"] == "detector":
            det.append((r["model"], float(r["median_latency_seconds"]), y))
        elif r["category"] == "baseline":
            ref_rec = (float(r["throughput_docs_per_second"]), y)
            ref_det = (float(r["median_latency_seconds"]), y)
    fig02a(rec, ref_rec)
    fig02b(det, ref_det)


# --- 04: batch sweep ---------------------------------------------------------

def load_batch_summary():
    return json.load(open(ROOT / "outputs/benchmarks/09.batch-size-sweep/20260822T140836Z/summary.json"))


def fig04a(rows):
    sweep = defaultdict(dict)
    for r in rows:
        if r.get("changed_stage") not in ("text_detection", "text_recognition"):
            continue
        key = (r["changed_stage"], r["document_type"])
        sweep[key][r["requested_batch_size"]] = (
            r["median_text_detection_seconds"] if r["changed_stage"] == "text_detection"
            else r["median_text_recognition_seconds"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), dpi=150,
                             facecolor=SURFACE, layout="constrained")
    for ax, stage, sel, sname in ((axes[0], "text_detection", 1, "Detection stage"),
                                  (axes[1], "text_recognition", 2, "Recognition stage")):
        for doc in ("passport", "id_card", "driving_license"):
            pts = sorted(sweep[(stage, doc)].items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5,
                    lw=2, color=DOC_COLORS[doc], label=DOC_LABELS[doc])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32])
        ax.set_xlabel("Requested stage batch size")
        ax.set_title(sname, loc="left", fontsize=10.5, fontweight="bold")
        ax.axvline(sel, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.text(sel * 1.12, 0.05, f"selected: {sel}", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK_2, ha="left", va="bottom")
        style(ax)
    axes[0].set_ylabel("Median stage seconds (20-doc corpus)")
    legend(axes[0], loc="upper center", bbox_to_anchor=(1.0, -0.16), ncol=3)
    save(fig, "04_batch_stage_sweep.png")


def combined_label(configuration):
    m = re.match(r"localization_(\d+)_detection_(\d+)_recognition_(\d+)", configuration)
    return f"L{m.group(1)}-D{m.group(2)}-R{m.group(3)}" if m else configuration


# Label offsets (points) for the combined-config scatter, keyed by config label.
FIG04B_OFFSETS = {
    "L4-D1-R2": (10, -4), "L4-D1-R4": (10, 0), "L4-D1-R8": (10, 6),
    "L4-D1-R32": (10, 0), "L4-D8-R2": (8, 6), "L4-D8-R4": (-8, 8),
    "L4-D16-R2": (8, -12), "L1-D8-R2": (10, -2),
}


def fig04b(rows):
    combined = defaultdict(lambda: {"lat": [], "rss": 0.0})
    for r in rows:
        if "changed_stage" in r and r["changed_stage"] in (
                "text_detection", "text_recognition"):
            continue
        c = combined[combined_label(r["configuration"])]
        c["lat"].append(float(r["median_total_latency_seconds"]))
        c["rss"] = max(c["rss"], float(r["median_peak_rss_mb"]))
    pts = sorted((np.mean(v["lat"]), v["rss"], k) for k, v in combined.items())
    fig, ax = new_fig(7.0, 4.2)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=52, color=CANDIDATE,
               zorder=2)
    best = next(p for p in pts if p[2] == "L4-D1-R2")
    for x, y, name in pts:
        dx, dy = FIG04B_OFFSETS.get(name, (8, 4))
        ha = "left" if dx > 0 else "right"
        label = f"{name} · selected" if name == "L4-D1-R2" else name
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color=INK_2, ha=ha)
    highlight(ax, best[0], best[1], SELECTED)
    ax.set_xlabel("Mean median total latency across the three routes (s) — lower is better")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_xlim(min(p[0] for p in pts) - 0.08, max(p[0] for p in pts) + 0.08)
    ax.set_ylim(min(p[1] for p in pts) - 60, max(p[1] for p in pts) + 60)
    style(ax)
    save(fig, "04_batch_combined_tradeoff.png")


def fig04():
    rows = load_batch_summary()
    fig04a(rows)
    fig04b(rows)


# --- 05: recognizer A/B ------------------------------------------------------

def fig05a():
    data = json.load(open(ROOT / "outputs/benchmarks/10.recognizer-a-b-comparison/20260822T142128Z/summary.json"))
    runs = defaultdict(list)
    for r in data["runs"]:
        runs[r["recognizer"]].append(r)
    order = ["PP-OCRv6_medium_rec", "latin_PP-OCRv5_mobile_rec"]
    labels = ["PP-OCRv6\nmedium", "Latin v5\nmobile"]
    fig, ax = new_fig(7.4, 2.9)
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for key, label in [("localization", "Localization"), ("text_detection", "Detection"),
                       ("text_recognition", "Recognition")]:
        vals = np.array([median(r["stages"][key] for r in runs[rec]) for rec in order])
        ax.barh(y, vals, 0.45, left=left, color=STAGE_COLORS[label],
                label=label, edgecolor=SURFACE, linewidth=2)
        left += vals
    other = np.array([median(r["other_seconds"] + r["stages"].get("pipeline", 0)
                             for r in runs[rec]) for rec in order])
    ax.barh(y, other, 0.45, left=left, color=STAGE_COLORS["Other"], label="Other",
            edgecolor=SURFACE, linewidth=2)
    left += other
    for yi_pos, rec in enumerate(order):
        tot = median(r["total_seconds"] for r in runs[rec])
        ax.text(left[yi_pos] + 0.6, y[yi_pos], f"{tot:.1f} s", va="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Median full-corpus latency (seconds)")
    ax.set_xlim(0, 58)
    ax.set_ylim(-0.55, 1.75)
    style(ax, ygrid=False)
    legend(ax, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save(fig, "05_recognizer_ab_stage_timing.png")


def fig05b():
    ab = json.load(open(ROOT / "outputs/benchmarks/10.recognizer-a-b-comparison/20260822T142128Z/summary.json"))
    runs = defaultdict(list)
    for r in ab["runs"]:
        runs[r["recognizer"]].append(r)
    hist = {r["model"]: r["median_latency_seconds"] for r in json.load(open(
        ROOT / "outputs/benchmarks/08.recognizer-final-confirmation/20260822T181139Z/aggregate.json"))}
    order = ["PP-OCRv6_medium_rec", "latin_PP-OCRv5_mobile_rec"]
    labels = ["PP-OCRv6 medium", "Latin v5 mobile"]
    fig, ax = new_fig(7.0, 2.6)
    y = np.arange(len(order))[::-1]
    for yi, rec, lab in zip(y, order, labels):
        rep = median(r["total_seconds"] for r in runs[rec])
        h = hist[rec]
        ax.plot([h, rep], [yi, yi], color=GRID, lw=3, zorder=1)
        ax.scatter([h], [yi], s=70, facecolor=SURFACE, edgecolor=CANDIDATE,
                   linewidth=2, zorder=2, label="historical run" if yi == 1 else None)
        ax.scatter([rep], [yi], s=70, color=CANDIDATE, zorder=3,
                   label="reproduced (median of 2)" if yi == 1 else None)
        ax.annotate(f"{h:.1f} s", (h, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color=INK_2)
        ax.annotate(f"{rep:.1f} s", (rep, yi), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Full-corpus latency (seconds)")
    ax.set_xlim(10, 58)
    ax.set_ylim(-0.7, 1.7)
    style(ax)
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, "05_recognizer_reproduction.png")


def fig05():
    fig05a()
    fig05b()


# --- 06: runtime and preprocessing -------------------------------------------

def fig06a():
    rows = read_csv(ROOT / "outputs/benchmarks/11.cpu-thread-count-benchmark/comparison.csv")
    series = defaultdict(list)
    for r in rows:
        series[r["document_type"].replace("-", "_")].append(
            (int(r["thread_count"]), float(r["e2e_seconds_median"])))
    fig, ax = new_fig(7.0, 4.2)
    for doc, pts in series.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, lw=2,
                color=DOC_COLORS[doc], label=DOC_LABELS[doc])
    ax.axvline(4, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(4.2, 0.94, "selected: 4", transform=ax.get_xaxis_transform(),
            fontsize=8, color=INK_2)
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Median end-to-end latency (seconds)")
    ax.set_xticks([1, 2, 4, 6, 8, 12, 16])
    style(ax)
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    save(fig, "06_cpu_thread_scaling.png")


def fig06b():
    run_dir = ROOT / "outputs/benchmarks/12.recognition-packing-comparison/20260824T075726Z"
    rows = read_csv(run_dir / "aggregate.csv")
    diffs = json.load(open(run_dir / "output_differences.json"))
    strat_label = {"current": "Current\nproduction", "aspect-ratio": "Aspect\nratio",
                   "fixed-width-buckets": "Fixed-width\nbuckets", "best-fit": "Best fit"}
    strats = ["current", "aspect-ratio", "fixed-width-buckets", "best-fit"]
    docs = ["passport", "id_card", "driving_license"]
    doc_runs = {"passport": 27, "id_card": 12, "driving_license": 21}
    sem = defaultdict(int)
    for e in diffs:
        sem[(e["strategy"], e["document_type"])] += 1
    agg = defaultdict(list)
    for r in rows:
        agg[(r["strategy"], r["document_type"])].append(float(r["e2e_seconds"]))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), dpi=150, facecolor=SURFACE,
                             layout="constrained", sharey=False)
    for ax, doc in zip(axes, docs):
        vals = [median(agg[(s, doc)]) for s in strats]
        bars = ax.bar(range(4), vals, 0.6, color=FOUR_COLORS,
                      edgecolor=SURFACE, linewidth=1)
        for i, (b, s) in enumerate(zip(bars, strats)):
            note = "reference" if s == "current" else \
                f"{doc_runs[doc] - sem[(s, doc)]} / {sem[(s, doc)]}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, note,
                    ha="center", fontsize=8, color=INK_2)
        ax.set_xticks(range(4), [strat_label[s] for s in strats], fontsize=7.8)
        ax.set_title(DOC_LABELS[doc], fontsize=10.5, fontweight="bold", loc="left")
        ax.set_ylim(0, max(vals) * 1.22)
        style(ax)
    axes[0].set_ylabel("Median end-to-end latency (s)")
    save(fig, "06_recognition_packing.png")


def fig06c():
    rows = read_csv(ROOT / "outputs/benchmarks/13.detector-resolution-workload/20260824T090238Z/broad_sweep.csv")
    time_series, change_series = defaultdict(list), defaultdict(list)
    for r in rows:
        doc = r["document_type"].replace("-", "_")
        px = float(r["actual_pixel_ratio_vs_baseline_pct"])
        time_series[doc].append((px, float(r["median_detection_seconds"])))
        change_series[doc].append((px, int(r["output_differences_vs_100"])))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), dpi=150, facecolor=SURFACE,
                             layout="constrained")
    for ax, series, ylab in (
            (axes[0], time_series, "Median detection seconds"),
            (axes[1], change_series, "Output changes vs 100% baseline")):
        for doc, pts in series.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, lw=2,
                    color=DOC_COLORS[doc], label=DOC_LABELS[doc])
        ax.axvline(100, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.set_xlabel("Actual detector pixels vs baseline (%)")
        ax.set_ylabel(ylab)
        style(ax)
    axes[0].set_title("Cost: detection time", loc="left", fontsize=10.5, fontweight="bold")
    axes[1].set_title("Risk: changed outputs", loc="left", fontsize=10.5, fontweight="bold")
    legend(axes[0], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    save(fig, "06_detector_resolution.png")


def fig06d():
    rows = json.load(open(ROOT / "outputs/benchmarks/14.mrz-preprocessing-reconciliation/"
                               "20260824T111635Z/mrz_aggregate.json"))
    order = ["original", "contrast_1.30", "contrast_1.50", "sharpen_light"]
    labels = ["Original\n(baseline)", "Contrast\n1.30", "Contrast 1.50\n(candidate)",
              "Light\nsharpening"]
    by = {r["variant"]: r for r in rows}
    panels = [
        ("Exact MRZ documents", [by[v]["total_exact"] for v in order], 13, False),
        ("Exact MRZ lines", [by[v]["exact_lines"] for v in order], 30, False),
        ("MRZ character errors", [by[v]["mrz_character_errors"] for v in order], None, True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), dpi=150, facecolor=SURFACE,
                             layout="constrained")
    for ax, (heading, vals, denom, lower_better) in zip(axes, panels):
        bars = ax.bar(range(4), vals, 0.6, color=FOUR_COLORS, edgecolor=SURFACE, linewidth=1)
        for i, b in enumerate(bars):
            note = f"{vals[i]}" + (f"/{denom}" if denom else "")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.03, note,
                    ha="center", fontsize=8.5, color=INK, fontweight="bold")
        ax.set_xticks(range(4), labels, fontsize=7.8)
        ax.set_title(heading + ("  (lower is better)" if lower_better else ""),
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_ylim(0, max(vals) * 1.25)
        style(ax)
    save(fig, "06_mrz_preprocessing.png")


def fig06():
    fig06a()
    fig06b()
    fig06c()
    fig06d()


# --- 07: final Latin-vs-current verification comparison ---------------------

VERIFICATION_ROOT = ROOT / "outputs/benchmarks/22.latin-vs-current-matcher/20260902T112500Z"

def verification_rows(name):
    return read_csv(VERIFICATION_ROOT / name)


def fig07a():
    rows = verification_rows("document_type_accuracy.csv")
    docs = list(DOC_LABELS)
    by_doc = {(r["candidate"], r["document_type"]): r for r in rows}
    x = np.arange(len(docs))
    width = 0.36
    fig, ax = new_fig(7.2, 4.1)
    for offset, candidate, label, color in [(-width / 2, "LATIN_OLD", "Latin old", "#6f6e69"),
                                             (width / 2, "MATCHING_NEW", "Matching new", "#eb6834")]:
        vals = [float(by_doc[(candidate, d)]["accuracy"]) * 100 for d in docs]
        bars = ax.bar(x + offset, vals, width, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1,
                    f"{value:.1f}%", ha="center", va="bottom", fontsize=8,
                    color=INK_2)
    ax.set_xticks(x, [DOC_LABELS[d] for d in docs])
    ax.set_ylabel("Accepted field accuracy (%)")
    ax.set_ylim(0, 108)
    style(ax)
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save(fig, "07_verification_accuracy.png")


def fig07b():
    rows = verification_rows("failure_transitions.csv")
    grouped = {}
    for r in rows:
        key = (r["document_type"], r["field"])
        item = grouped.setdefault(key, {"latin": 0, "current": 0})
        item["latin"] += r["latin_status"] in {"mismatch", "not_found"}
        item["current"] += r["current_status"] in {"mismatch", "not_found"}
    rows = [{"document_type": k[0], "field": k[1], **v} for k, v in grouped.items()]
    rows = [r for r in rows if r["latin"] or r["current"]]
    rows.sort(key=lambda r: (-max(r["latin"], r["current"]), r["document_type"], r["field"]))
    labels = [f"{r['document_type']}/{r['field']}" for r in rows]
    y = np.arange(len(rows))
    fig, ax = new_fig(8.8, 6.5)
    height = 0.34
    base = [r["latin"] for r in rows]
    cur = [r["current"] for r in rows]
    bars_base = ax.barh(y - height / 2, base, height, color="#6f6e69", label="Latin old")
    bars_cur = ax.barh(y + height / 2, cur, height, color="#eb6834", label="Matching new")
    ax.bar_label(bars_base, fmt="%d", padding=3, fontsize=8, color=INK_2)
    ax.bar_label(bars_cur, fmt="%d", padding=3, fontsize=8, color=INK_2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Hard failures (unique fields)")
    ax.set_xlim(0, max(base + cur) + 1.0)
    style(ax)
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save(fig, "07_verification_hard_failures.png")


def fig07c():
    rows = verification_rows("raw_runs.csv")
    vals = {candidate: [float(r["end_to_end_seconds"])
                        for r in rows if r.get("candidate") == candidate and r.get("end_to_end_seconds")]
            for candidate in ("LATIN_OLD", "MATCHING_NEW")}
    fig, ax = new_fig(7.2, 4.1)
    bp = ax.boxplot([vals["LATIN_OLD"], vals["MATCHING_NEW"]], patch_artist=True,
                    tick_labels=["Latin old", "Matching new"], widths=0.42,
                    boxprops={"edgecolor": INK_2},
                    medianprops={"color": INK, "linewidth": 1.2},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                    flierprops={"marker": "o", "markerfacecolor": SURFACE,
                                "markeredgecolor": INK, "markersize": 5})
    for box, color in zip(bp["boxes"], ["#6f6e69", "#eb6834"]):
        box.set_facecolor(color)
        box.set_alpha(0.9)
    for position, candidate in enumerate(("LATIN_OLD", "MATCHING_NEW"), start=1):
        ax.text(position, median(vals[candidate]), f"median {median(vals[candidate]):.2f}s",
                ha="center", va="bottom", fontsize=8, color=INK_2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_ylabel("End-to-end seconds/document")
    style(ax, ygrid=False)
    save(fig, "07_verification_latency.png")


def fig07d():
    rows = [r for r in verification_rows("stage_timings.csv")
            if r["stage"] in {"text_detection", "text_recognition", "verification"}]
    stages = [("text_detection", "Detection"), ("text_recognition", "Recognition"),
              ("verification", "Verification")]
    candidates = [("LATIN_OLD", "Latin old", "#6f6e69"),
                  ("MATCHING_NEW", "Matching new", "#eb6834")]
    x = np.arange(len(stages))
    width = 0.36
    fig, ax = new_fig(7.6, 4.2)
    by = {(r["candidate"], r["stage"]): float(r["median_ms_per_doc"]) for r in rows}
    for offset, candidate, label, color in [(-width / 2, *candidates[0]), (width / 2, *candidates[1])]:
        vals = [by[(candidate, stage)] for stage, _ in stages]
        bars = ax.bar(x + offset, vals, width, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 8,
                    f"{value:.0f}", ha="center", va="bottom", fontsize=8,
                    color=INK_2)
    ax.set_xticks(x, [label for _, label in stages])
    ax.set_ylabel("Median milliseconds/document")
    ax.set_ylim(0, max(float(r["median_ms_per_doc"]) for r in rows) * 1.22)
    style(ax)
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save(fig, "07_verification_stage_share.png")


def fig07():
    fig07a()
    fig07b()
    fig07c()
    fig07d()


# --- 08: full pipeline vs direct matching, exact-vs-exact -------------------

ARCH_ROOT = ROOT / "outputs/benchmarks/20.full-pipeline-vs-direct-matching/20260902T071921Z"
ARCH_AUDIT_ROOT = ROOT / "outputs/benchmarks/21.full-pipeline-vs-direct-matching-audit/20260902T080411Z"
ARCH_FULL = "FULL_LATIN_PIPELINE"
ARCH_DIRECT = "DIRECT_MATCHING_PIPELINE"


def _architecture_rows(name):
    return read_csv(ARCH_ROOT / name)


def fig08a():
    rows = _architecture_rows("document_type_accuracy.csv")
    docs = ["passport", "id_card", "driving_license"]
    by = {(r["candidate"], r["document_type"]): r for r in rows}
    x = np.arange(len(docs))
    width = 0.36
    fig, ax = new_fig(7.4, 4.2)
    full_vals = [100 * float(by[(ARCH_FULL, doc)]["accuracy"]) for doc in docs]
    bars = ax.bar(x - width / 2, full_vals, width, color="#6f6e69", label="Full exact",
                  edgecolor=SURFACE, linewidth=1)
    for bar, value in zip(bars, full_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=8, color=INK_2)

    direct_exact = [100 * float(by[(ARCH_DIRECT, doc)]["accuracy"]) for doc in docs]
    direct_likely = [100 * int(by[(ARCH_DIRECT, doc)]["matcher_likely_match"]) /
                     int(by[(ARCH_DIRECT, doc)]["unique_evaluated_fields"]) for doc in docs]
    exact_bars = ax.bar(x + width / 2, direct_exact, width, color=ARCH_STAGE_COLORS["Matching"],
                        label="Direct exact", edgecolor=SURFACE, linewidth=1)
    likely_bars = ax.bar(x + width / 2, direct_likely, width, bottom=direct_exact,
                         color=ARCH_STAGE_COLORS["MRZ"], label="Direct likely (not counted)",
                         edgecolor=SURFACE, linewidth=1)
    for bar, value in zip(exact_bars, direct_exact):
        ax.text(bar.get_x() + bar.get_width() / 2, value - 3.0, f"{value:.1f}%",
                ha="center", va="center", fontsize=8, color="#ffffff", fontweight="bold")
    for index, (bar, value, count) in enumerate(zip(
            likely_bars, direct_likely,
            [int(by[(ARCH_DIRECT, doc)]["matcher_likely_match"]) for doc in docs])):
        if count:
            inside = value >= 4
            ax.text(bar.get_x() + bar.get_width() / 2,
                    direct_exact[index] + value / 2 if inside else direct_exact[index] + value + 1.1,
                    f"+{count}", ha="center", va="center" if inside else "bottom", fontsize=8,
                    color="#ffffff" if inside else ARCH_STAGE_COLORS["MRZ"],
                    fontweight="bold")
    ax.set_xticks(x, [DOC_LABELS[doc] for doc in docs])
    ax.set_ylabel("Exact field accuracy (%)")
    ax.set_ylim(0, 108)
    style(ax)
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    save(fig, "08_accuracy_comparison.png")


def fig08b():
    accuracy = {r["candidate"]: r for r in _architecture_rows("overall_accuracy.csv")}
    performance = {r["candidate"]: r for r in _architecture_rows("performance.csv")}
    points = [(ARCH_FULL, "Full Latin pipeline", "#6f6e69"),
              (ARCH_DIRECT, "Direct matching pipeline", ARCH_STAGE_COLORS["Matching"])]
    xs = [float(performance[c]["median_ms_per_doc"]) for c, _, _ in points]
    ys = [100 * float(accuracy[c]["accuracy"]) for c, _, _ in points]
    fig, ax = new_fig(7.3, 4.5)
    ax.plot(xs, ys, color=GRID, linewidth=2, linestyle=(0, (4, 3)), zorder=1)
    for (candidate, label, color), x, y in zip(points, xs, ys):
        ax.scatter([x], [y], s=90, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.annotate(f"{label}\n{y:.2f}% · {x:.0f} ms",
                    (x, y), textcoords="offset points",
                    xytext=(10, 10 if candidate == ARCH_FULL else -31),
                    fontsize=8.5, color=INK_2)
    ax.text(0.03, 0.06, "Pareto frontier", transform=ax.transAxes, fontsize=8.5, color=INK_2)
    ax.set_xlabel("Median latency (ms/document) — lower is better")
    ax.set_ylabel("Exact field accuracy (%) — higher is better")
    ax.set_ylim(65, 100)
    style(ax)
    save(fig, "08_pareto_frontier.png")


def fig08c():
    rows = [r for r in read_csv(ARCH_AUDIT_ROOT / "timing_accounting.csv")
            if r["row_type"] == "aggregate" and r["statistic"] == "mean"]
    by = {r["candidate"]: r for r in rows}
    stages = ["Localization", "MRZ", "Canonicalization + crop", "Detection",
              "Recognition", "Matching", "Other"]
    def other(row):
        return sum(float(row[name]) for name in (
            "input_preparation", "text_line_cropping", "ocr_unpacking",
            "result_assembly", "other"))
    values = {
        ARCH_FULL: [float(by[ARCH_FULL]["localization"]), float(by[ARCH_FULL]["mrz_total"]),
                    float(by[ARCH_FULL]["canonicalization_crop"]), float(by[ARCH_FULL]["detection"]),
                    float(by[ARCH_FULL]["recognition"]), float(by[ARCH_FULL]["matching_extraction"]),
                    other(by[ARCH_FULL])],
        ARCH_DIRECT: [0, 0, 0, float(by[ARCH_DIRECT]["detection"]),
                      float(by[ARCH_DIRECT]["recognition"]), float(by[ARCH_DIRECT]["matching_extraction"]),
                      other(by[ARCH_DIRECT])],
    }
    fig, ax = new_fig(8.0, 4.3)
    y = np.arange(2)
    left = np.zeros(2)
    for index, stage in enumerate(stages):
        vals = np.array([values[ARCH_FULL][index], values[ARCH_DIRECT][index]])
        ax.barh(y, vals, 0.48, left=left, color=ARCH_STAGE_COLORS[stage], label=stage,
                edgecolor=SURFACE, linewidth=1.5)
        for yi, value, start in zip(y, vals, left):
            if value >= 35:
                ax.text(start + value / 2, yi, f"{value:.0f}", ha="center", va="center",
                        fontsize=8, color="#ffffff", fontweight="bold")
        left += vals
    totals = [float(by[ARCH_FULL]["total"]), float(by[ARCH_DIRECT]["total"])]
    for yi, total in zip(y, totals):
        ax.text(total + 25, yi, f"{total:.0f} ms", va="center", fontsize=9,
                color=INK, fontweight="bold")
    ax.set_yticks(y, ["Full Latin pipeline", "Direct matching pipeline"])
    ax.set_xlabel("Mean request time (ms/document)")
    ax.set_xlim(0, max(totals) * 1.13)
    style(ax, ygrid=False)
    legend(ax, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    save(fig, "08_stage_breakdown.png")


def fig08d():
    detection = _architecture_rows("detection_workload.csv")
    recognition = _architecture_rows("recognition_workload.csv")
    values = {
        "Detector tensor pixels": {
            ARCH_FULL: median(int(r["total_detector_pixels"]) for r in detection if r["candidate"] == ARCH_FULL),
            ARCH_DIRECT: median(int(r["total_detector_pixels"]) for r in detection if r["candidate"] == ARCH_DIRECT),
        },
        "Recognition crops": {
            ARCH_FULL: median(int(r["recognition_crops_per_document"]) for r in recognition if r["candidate"] == ARCH_FULL),
            ARCH_DIRECT: median(int(r["recognition_crops_per_document"]) for r in recognition if r["candidate"] == ARCH_DIRECT),
        },
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), dpi=180, facecolor=SURFACE,
                             layout="constrained")
    for ax, metric, heading in zip(axes, values, ["Detector tensor pixels", "Recognition crops"]):
        vals = [values[metric][ARCH_FULL], values[metric][ARCH_DIRECT]]
        bars = ax.bar([0, 1], vals, 0.56, color=["#6f6e69", ARCH_STAGE_COLORS["Matching"]],
                      edgecolor=SURFACE, linewidth=1)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.03, f"{value:,.0f}",
                    ha="center", fontsize=8.5, color=INK_2)
        ax.set_title(heading, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xticks([0, 1], ["Full", "Direct"])
        ax.set_ylim(0, max(vals) * 1.18)
        style(ax)
    axes[0].set_ylabel("Median pixels/document")
    axes[1].set_ylabel("Median crops/document")
    save(fig, "08_workload_comparison.png")


def fig08e():
    rows = _architecture_rows("raw_runs.csv")
    values = [[float(r["server_ms"]) for r in rows if r["candidate"] == candidate]
              for candidate in (ARCH_FULL, ARCH_DIRECT)]
    fig, ax = new_fig(7.2, 4.1)
    bp = ax.boxplot(values, patch_artist=True,
                    tick_labels=["Full Latin pipeline", "Direct matching pipeline"], widths=0.42,
                    boxprops={"edgecolor": INK_2}, medianprops={"color": INK, "linewidth": 1.2},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2},
                    flierprops={"marker": "o", "markerfacecolor": SURFACE,
                                "markeredgecolor": INK, "markersize": 5})
    for box, color in zip(bp["boxes"], ["#6f6e69", ARCH_STAGE_COLORS["Matching"]]):
        box.set_facecolor(color)
        box.set_alpha(0.9)
    for position, data in enumerate(values, start=1):
        ax.text(position, median(data), f"p50 {median(data):.0f}", ha="center", va="bottom",
                fontsize=8.5, color=INK_2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_ylabel("Server latency (ms/document)")
    style(ax, ygrid=False)
    save(fig, "08_latency_throughput.png")


def fig08():
    fig08a()
    fig08b()
    fig08c()
    fig08d()
    fig08e()


def main():
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()


if __name__ == "__main__":
    main()
