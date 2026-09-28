# Low-Light Mosquito Detection and Classification

Detect and classify mosquitoes in phone-captured images into six classes, scored by **mAP**.
Approach: **Faster R-CNN (ResNet-50 + FPN)** fine-tuned from COCO weights, with CLAHE + gamma
enhancement on dark images.

| Class | id |
|---|---|
| aegypti | 0 |
| albopictus | 1 |
| anopheles | 2 |
| culex | 3 |
| culiseta | 4 |
| japonicus/koreicus | 5 |

Data: 7500 training images (YOLO-format labels: `class xc yc w h`, normalized) and 525 test images.

## Layout

```
configs/default.yaml     hyperparameters and paths
src/mosquito/            config, low-light enhancement, dataset, model, train/eval engine, submission
scripts/train.py         train, keep best checkpoint, tune score threshold
scripts/predict.py       test inference, writes and validates submission.csv
notebooks/               original exploratory notebook (reset50fast-rnn.ipynb)
tests/                   unit tests (no GPU or data needed)
data/                    put the competition data here (git-ignored; see data/README.md)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# put the data under data/ (see data/README.md), then:
python scripts/train.py --epochs 10
python scripts/predict.py            # -> outputs/submission.csv
pytest
```

Edit `configs/default.yaml` to change settings; `train.py` also accepts
`--data-root`, `--out-dir`, `--epochs`, `--batch-size`, `--lr`.

On Kaggle: pass `--data-root /kaggle/input/<competition>/<data folder>` and
`--out-dir /kaggle/working`, and set `sample_submission` in the config to the competition's
`sample_submission.csv`.

## Outputs (`outputs/`)

`fasterrcnn_best.pt` (best val mAP50-95), `training_history.csv`, `threshold_sweep.csv`,
`best_threshold.json`, `submission.csv`.

## Submission format

`id, ImageID, LabelName, Conf, xcenter, ycenter, bbx_width, bbx_height`: one row per image
(top detection), same row order as `sample_submission.csv`. Images with no detection above the
threshold get a low-confidence default box. `predict.py` validates columns, row order, labels
and value ranges before saving.

## Changes from the notebook

- Split into an installable package with scripts and a YAML config.
- Added horizontal-flip augmentation (the notebook's `train` flag was unused).
- `submission.csv` is written to `outputs/` (the notebook saved it to the working directory
  while printing a different path).
- The score threshold is tuned during training and reused at prediction time.

## Ideas to improve

- The metric is mAP but the submission keeps one box per image. Check whether the sample
  submission allows multiple rows per image; if so, keeping more detections can raise mAP.
- Higher-resolution input or FPN tuning for small mosquitoes; ResNet-101 backbone.
- Compare against a YOLO model on the same validation split.
- Stronger augmentation (brightness/contrast jitter) for low-light robustness.
