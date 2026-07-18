# Notebooks

## `train_yolo11_colab.ipynb`

Trains and **compares YOLO11 sizes (n / s / m / l)** for **Model A** (detect: person, vehicle)
or **Model B** (segment: building_damaged, road_blocked, water) on Google Colab's free GPU,
then saves weights to Google Drive for local use in Phase 3.

Only the model size changes across runs — data, image size, epochs and seed are identical, so
the comparison is a fair speed-vs-accuracy study. It prints a table (params, mAP@50,
mAP@50-95, inference ms, train time) and saves it to Drive as `compare_<task>.csv`.

**Workflow (Model A):**

1. Build + zip the dataset locally:
   ```bash
   make build-datasets ARGS="--sources visdrone sard --copy-images"
   (cd data/unified && zip -r -0 detect.zip detect)      # ~2.8 GB, store mode
   ```
2. Upload `detect.zip` to Google Drive (e.g. `MyDrive/uav/detect.zip`).
3. Open the notebook in Colab, set **Runtime → GPU**, run all cells.
4. Read the comparison table, then run cell 7 to retrain the winning size for 100 epochs.
5. Download `best.pt` → repo `weights/model_a.pt`.

For **Model B**: build the seg set (`--sources rescuenet floodnet --copy-images
--max-image-side 1280`), zip `seg`, and in the config cell set `TASK='segment'`,
`DATASET_ZIP=.../seg.zip`, `IMGSZ=1024`.

Hyperparameters mirror `configs/perception/model_a.yaml` / `model_b.yaml` — keep them in sync.
The sweep uses 50 epochs (cheap, fair); the winner is retrained at 100. On the free T4, the
`l` size at large image size is heavy — drop to `['n','s','m']` or split tasks across sessions
if you run out of GPU time. Weights (`*.pt`) are gitignored; store them under `weights/`.
