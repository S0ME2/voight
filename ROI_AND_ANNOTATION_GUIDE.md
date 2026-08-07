# ROI and annotation maintenance

`annotations/annotation_state.json` is the single source of truth for every
supported document: passport, ID card, and driving licence. Do not edit runtime
ROI files directly. They are generated from that state.

## Inputs and layout definitions

The input folders and annotation behaviour are defined in
`config/annotation_layouts.json`:

```text
annotation_input/
├── passports/
├── id_cards/<pair>/front.<image>
├── id_cards/<pair>/back.<image>
└── driving_licenses/
```

The supplied canonical driving-licence sample is in
`annotation_input/driving_licenses/`. Add another supported layout only by
updating `config/annotation_layouts.json` first: it specifies its input folder,
coordinate mode, and canonical size. Field names are entered after each ROI is
drawn; they are deliberately not preserved in this layout definition.

## Annotating

Run the one annotator:

```bash
uv run python scripts/annotate.py annotation_input annotations
```

- Passport: the tool detects the MRZ, shows the detected MRZ overlay, and asks
  for the four page corners. It then rectifies the page and all data/field ROIs
  are drawn on that MRZ-anchored page. This makes page location independent of
  scans with two pages, extra margins, or crops.
- ID card: click the card corners, then draw the data crop and configured
  field ROIs.
- Driving licence: its input is already canonical, so draw only its data crop
  and field ROIs. Its generated runtime files remain
  `config/driving_license/data_crop.json` and
  `config/driving_license/field_rois_crop.json`; they are outputs, not another
  annotation source.

The saved passport `mrz_anchor` is inside `annotation_state.json`. The legacy
`annotations/previews/passport_mrz.json` is only a generated diagnostic preview
and is no longer used as configuration.

To revise a completed sample, rerun the command and select its displayed
number. Enter the field name after each drawn rectangle; it is stored with the
sample in `annotation_state.json`.

## Direct JSON edits

For small adjustments, back up and edit `annotations/annotation_state.json`:

```bash
cp annotations/annotation_state.json annotations/annotation_state.json.bak
```

`data_crop` is normalized to the annotation image and `fields.<name>` is
normalized inside `data_crop`. Bounds are always `0 <= x1 < x2 <= 1` and
`0 <= y1 < y2 <= 1`.

For a newly MRZ-anchored passport, the annotation image is the rectified page;
leave `mrz_anchor.mrz_polygon`, `mrz_anchor.page_corners`, and
`mrz_anchor.page_corners_relative_to_mrz_width` together. Do not replace them
with source-image hard-coded page coordinates.

## Validate and promote

After any change, run:

```bash
uv run python scripts/annotate.py annotation_input annotations --check
uv run python scripts/promote_annotations.py
MODEL_DIR= uv run python -m unittest discover -s tests -v
```

Validation regenerates annotation previews and reports. Promotion regenerates
the passport/ID-card profiles and both driving-licence runtime ROI files from
the same state file. Finally, start the CPU service with `make docker-up-d` and
test through `http://localhost:8000/docs`.
