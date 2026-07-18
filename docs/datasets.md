# Datasets

Provenance, licences and the exact class mapping applied to each source dataset. This is
dissertation material — kept precise and cited. The mapping here is the human-readable
counterpart of `configs/datasets/taxonomy.yaml` (the machine-readable source of truth);
totality is enforced by `tests/test_taxonomy.py`.

All datasets are **obtained separately** and stored under `data/raw/` (gitignored). Nothing
is fetched at run time (CLAUDE.md non-goal).

## Unified taxonomy (ADR-002)

Two models, two label types — never merged.

| Model | Task | Unified classes |
|---|---|---|
| **A** | detect (boxes) | `person` (0), `vehicle` (1) |
| **B** | segment (polygons) | `building_damaged` (0), `road_blocked` (1), `water` (2) |

## Sources

### VisDrone-DET — Model A (detect)

| | |
|---|---|
| **Gives** | `person`, `vehicle` from UAV altitude |
| **Size / format** | 6,471 train / 548 val / 1,610 test-dev; YOLO format via Ultralytics `data="VisDrone.yaml"` (~2 GB) |
| **Native resolution** | ~2000×1500 (varies) |
| **Capture platform** | Consumer UAVs, multiple Chinese cities |
| **Licence** | Research use; see VisDrone challenge terms |
| **Note** | **68 % of objects are <32×32 px** — this is the small-object problem; SAHI is mandatory (ADR-002 / CLAUDE.md) |

Native classes (Ultralytics YOLO order; `ignored-regions` and `others` are removed during
YOLO export):

| Native | → Unified |
|---|---|
| pedestrian, people | `person` |
| car, van, truck, bus | `vehicle` |
| bicycle, tricycle, awning-tricycle, motor | *dropped* |

### SARD — Model A (detect)

| | |
|---|---|
| **Gives** | `person` in injured/exhausted poses (best survivor proxy) |
| **Size / format** | 1,981 images, 1920×1080, ~6,525 person instances; Pascal VOC XML |
| **Capture platform** | UAV, 35-min flight; actors simulate exhausted/injured people incl. lying prone |
| **Licence** | IEEE DataPort — academic use |
| **Note** | The `Corr` extension adds fog/snow/blur variants (not used yet) |

Native classes: every annotation is a **person**, sub-labelled by pose. All poses map to
`person`; the pose is preserved as per-detection metadata for stratified evaluation
(Phase 2), **not** as a class.

| Native pose | → Unified |
|---|---|
| Standing, Walking, Running, Sitting, Lying, Not-Defined | `person` (pose kept as metadata) |

### RescueNet — Model B (segment)

| | |
|---|---|
| **Gives** | `building_damaged`, `road_blocked`, `water` |
| **Size / format** | 4,494 images, 3000×4000, 11 mask classes |
| **Capture platform** | DJI Mavic Pro, post-Hurricane Michael (2018) |
| **Licence** | **CC BY-NC-ND** — non-commercial academic use only; **do not redistribute derived label files** |

Native classes (mask palette indices 0–10; verified against Chowdhury et al. 2023):

| Idx | Native | → Unified |
|---|---|---|
| 0 | Background | *dropped* |
| 1 | Water | `water` |
| 2 | Building-No-Damage | *dropped* |
| 3 | Building-Minor-Damage | *dropped* |
| 4 | Building-Major-Damage | `building_damaged` |
| 5 | Building-Total-Destruction | `building_damaged` |
| 6 | Vehicle | *dropped* † |
| 7 | Road-Clear | *dropped* |
| 8 | Road-Blocked | `road_blocked` |
| 9 | Tree | *dropped* |
| 10 | Pool | *dropped* |

† RescueNet `Vehicle` is a **mask**; Model A (vehicle) wants **boxes**. Emitting vehicle boxes
via mask→bbox is a documented future extension (PROJECT_PLAN §3.4) — currently dropped so no
annotation is silently lost.

### FloodNet — Model B (segment)

| | |
|---|---|
| **Gives** | flood-specific `building_damaged`, `road_blocked`, `water` |
| **Size / format** | ~2,343 images, 3000×4000, 10 mask classes |
| **Capture platform** | DJI Mavic Pro, post-Hurricane Harvey (Texas); same lab (BinaLab) as RescueNet |
| **Licence** | Community/academic use (FloodNet challenge terms) |

Native classes (mask palette indices 0–9; verified against Rahnemoonfar et al. 2021):

| Idx | Native | → Unified |
|---|---|---|
| 0 | Background | *dropped* |
| 1 | Building-Flooded | `building_damaged` |
| 2 | Building-Non-Flooded | *dropped* |
| 3 | Road-Flooded | `road_blocked` |
| 4 | Road-Non-Flooded | *dropped* |
| 5 | Water | `water` |
| 6 | Tree | *dropped* |
| 7 | Vehicle | *dropped* |
| 8 | Pool | *dropped* |
| 9 | Grass | *dropped* |

## Which variant to download

Each dataset ships several variants. You only need the one that matches our two tasks —
**boxes** for the detect model, **masks** for the segment model. Put each under the path in
`configs/datasets/default.yaml`.

| Dataset | Download this variant | Skip | Put in |
|---|---|---|---|
| VisDrone | **VisDrone2019-DET** (object detection), train + val | VID / MOT (video, tracking) | `data/raw/visdrone/` — or let Ultralytics auto-download it |
| SARD | **ORIGINAL** full-resolution **colour** images + box labels | pre-tiled (`640`, `tiles_3x3`, `3-d_tiles`) and grayscale variants | `data/raw/sard/` |
| RescueNet | the **semantic-segmentation** set with label masks (train/val/test) | — | `data/raw/rescuenet/` |
| FloodNet | **FloodNet-Supervised v1.0** (segmentation, has label masks) | the semi-supervised / VQA "Track 1" set | `data/raw/floodnet/` |

**Where to actually download (verified 2026-07-18):**

- **VisDrone-DET** — auto-downloads via Ultralytics, or [official site](http://aiskyeye.com/).
- **SARD** — [IEEE DataPort](https://ieee-dataport.org/documents/search-and-rescue-image-dataset-person-detection-sard) (also mirrored on Roboflow/Kaggle). If exporting from Roboflow, choose **ORIGINAL** size, **colour**, and **Pascal VOC XML** format — that keeps the pose sub-labels our loader stores as metadata.
- **RescueNet** — the GitHub repo holds **only code**; the actual images + masks are on **figshare**: [Springer Nature figshare collection 6647354](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354).
- **FloodNet** — [github.com/BinaLab/FloodNet-Supervised_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0) (the README links a Dropbox download). This is the segmentation set with masks — **not** the ~12 GB datasetninja "Track 1" set, which is the VQA/semi-supervised version we don't use.

Folder layout the loaders expect (`src/perception/datasets/loaders.py`):

- **VisDrone** — `.../labels/*.txt` (YOLO boxes) with a sibling `images/` folder.
- **SARD** — `*.xml` (Pascal VOC) with the image alongside.
- **RescueNet / FloodNet** — label masks named `*lab*.png` (each pixel value = a class index),
  with the original `.jpg` alongside.

Tip: VisDrone is the easy one — it auto-downloads during training. The other three are manual
(IEEE DataPort for SARD; BinaLab GitHub for RescueNet + FloodNet).

## Real instance counts (built 2026-07-19)

From the actual downloads via `make build-datasets ARGS=--dry-run`:

**Detect set (Model A)** — 9,000 images (train 7,858 / val 944 / test 198):

| class | VisDrone | SARD | total |
|---|--:|--:|--:|
| person | 120,365 | 6,532 | **126,897** |
| vehicle | 205,663 | 0 | **205,663** |

**Segment set (Model B)** — 6,387 images (train 5,040 / val 899 / test 448); instances are
polygons (connected mask regions), images downscaled to 1280 px long side:

| class | RescueNet | FloodNet | total |
|---|--:|--:|--:|
| building_damaged | 3,389 | 3,210 | **6,599** |
| road_blocked | 785 | 1,858 | **2,643** |
| water | 1,756 | 4,841 | **6,597** |

`road_blocked` is markedly rarer than the other two — worth noting as class imbalance in the
evaluation chapter.

### On-disk formats actually encountered (loaders handle these)

- **VisDrone** ships the **raw** annotation format (`annotations/*.txt`, comma-separated,
  absolute pixels, category id in field 6, ids 0–11) — *not* pre-converted YOLO. The loader
  parses it directly and normalises by image size.
- **SARD** (Roboflow VOC export) collapses all six poses to a single **`human`** class; the
  pose sub-labels are therefore not available from this download.
- **RescueNet** masks are single-channel **index masks** (values 0–10); originals live in the
  sibling `*-org-img/` folder.
- **FloodNet** ⚠️ — the file you download must be the **index masks + original images** from
  `FloodNet-Supervised_v1.0`. The separate **`ColorMasks-FloodNetv1`** distribution is RGB
  *visualisation* masks with **no original images**, and cannot be used to train — the loader
  raises a clear error if handed an RGB mask.

## Known limitations (for the write-up)

- **No survivor labels in disaster imagery.** No disaster dataset (RescueNet, FloodNet, xBD)
  labels people. `person` is therefore trained on SAR/ordinary imagery (SARD, VisDrone) and
  transferred into the disaster domain. The domain gap is **measured and reported**, not
  hidden (PROJECT_PLAN §3.1). SARD (posed injured/prone actors) is the closest proxy.
- **"Broken bridges" has no dataset.** Folded into `road_blocked`.
- **Mask palette order is assumed to match the source papers' class order.** The loaders read
  palette index → class from `taxonomy.yaml`; verify against the first real mask sample when
  the data lands, and correct the `native:` list if the on-disk palette differs (one-line fix,
  no code change).

## Build

```bash
make build-datasets ARGS=--dry-run   # per-class instance-count table for both sets
make build-datasets                  # write data/unified/{detect,seg}
```

Output: two Ultralytics-style datasets under `data/unified/` — `detect/` (YOLO boxes) and
`seg/` (YOLO-seg polygons), each with `images/`, `labels/` and a `data.yaml`.

## References

- Zhu et al. (2021), *Detection and Tracking Meet Drones Challenge*, TPAMI — VisDrone.
- Sambolek & Ivašić-Kos (2021), *Automatic Person Detection in SAR Operations Using Deep CNN
  Detectors*, IEEE Access — SARD.
- Chowdhury et al. (2023), *RescueNet: A High Resolution UAV Semantic Segmentation Dataset for
  Natural Disaster Damage Assessment*, Scientific Data 10 — arXiv:2202.12361.
- Rahnemoonfar et al. (2021), *FloodNet: A High Resolution Aerial Imagery Dataset for Post
  Flood Scene Understanding*, IEEE Access — arXiv:2012.02951.
