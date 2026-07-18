# Notebooks

## `train_yolo11_colab.ipynb`

Trains **Model A** (detect: person, vehicle) or **Model B** (segment: building_damaged,
road_blocked, water) on Google Colab's free GPU, and saves weights to your Google Drive for
local use in Phase 3.

**Workflow (Model A):**

1. Build + zip the dataset locally:
   ```bash
   make build-datasets ARGS="--sources visdrone sard --copy-images"
   cd data/unified && zip -r detect.zip detect      # ~2.8 GB
   ```
2. Upload `detect.zip` to Google Drive (e.g. `MyDrive/uav/detect.zip`).
3. Open the notebook in Colab, set **Runtime → GPU**, edit the config cell, run all cells.
4. Download `best.pt` from Drive → repo `weights/model_a.pt`.

For **Model B**, build the seg set instead (`--sources rescuenet floodnet`), zip `seg`, and
switch `MODEL`/`TASK`/`IMGSZ` in the config cell (values are in the notebook).

Hyperparameters mirror `configs/perception/model_a.yaml` and `model_b.yaml` — keep them in sync.
Weights (`*.pt`) are gitignored; store them under `weights/` locally.
