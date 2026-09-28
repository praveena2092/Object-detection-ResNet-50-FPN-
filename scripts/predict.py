"""Run the trained model on the test set and write submission.csv."""
import argparse
import glob
import json
import os

import torch

from mosquito.config import NUM_CLASSES, load_config
from mosquito.model import load_checkpoint
from mosquito.submission import finalize_submission, predict_test, validate_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--score-thresh", type=float, help="override; default = tuned value from train.py")
    args = ap.parse_args()

    cfg = load_config(args.config, data_root=args.data_root, out_dir=args.out_dir)
    ckpt = args.checkpoint or os.path.join(cfg.out_dir, "fasterrcnn_best.pt")
    thresh = args.score_thresh
    if thresh is None:
        tuned = os.path.join(cfg.out_dir, "best_threshold.json")
        thresh = json.load(open(tuned))["score_thresh"] if os.path.exists(tuned) else cfg.score_thresh
    print(f"Checkpoint: {ckpt} | score threshold: {thresh}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(ckpt, NUM_CLASSES + 1, device)
    test_images = sorted(glob.glob(os.path.join(cfg.test_img, "*")))
    print(f"Test images: {len(test_images)}")

    pred_df = predict_test(model, test_images, device, thresh, cfg.enhance_low_light)
    print(f"Real detections: {pred_df["class_id"].notna().sum()} / {len(test_images)}")
    sub = finalize_submission(pred_df, cfg.sample_submission)

    problems = validate_submission(sub, cfg.sample_submission)
    if problems:
        raise SystemExit("Submission invalid:\n  - " + "\n  - ".join(problems))
    out = os.path.join(cfg.out_dir, "submission.csv")
    sub.to_csv(out, index=False)
    print("Saved:", out)
    print(sub["LabelName"].value_counts())


if __name__ == "__main__":
    main()
