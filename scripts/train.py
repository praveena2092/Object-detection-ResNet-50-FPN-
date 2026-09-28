"""Train Faster R-CNN, keep the best-mAP checkpoint, and tune the score threshold."""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mosquito.config import NUM_CLASSES, load_config
from mosquito.dataset import MosquitoDataset, collate_fn, list_labeled_images, split_train_val
from mosquito.engine import evaluate_map, sweep_score_threshold, train_one_epoch
from mosquito.model import build_model, load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    args = ap.parse_args()

    cfg = load_config(args.config, data_root=args.data_root, out_dir=args.out_dir,
                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("Device:", device)

    labeled = list_labeled_images(cfg.train_img, cfg.train_lbl)
    train_paths, val_paths = split_train_val(labeled, cfg.val_fraction, cfg.seed)
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    train_ds = MosquitoDataset(train_paths, cfg.train_lbl, cfg.enhance_low_light, train=True, hflip=cfg.hflip)
    val_ds = MosquitoDataset(val_paths, cfg.train_lbl, cfg.enhance_low_light, train=False)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fn)

    model = build_model(NUM_CLASSES + 1).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(cfg.epochs * 0.6)), gamma=0.1)

    best, history = -1.0, []
    best_path = os.path.join(cfg.out_dir, "fasterrcnn_best.pt")
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg.epochs)
        scheduler.step()
        m = evaluate_map(model, val_loader, device)
        print(f"Epoch {epoch}: loss={loss:.4f} | val mAP50-95={m["map50_95"]:.4f} | mAP50={m["map50"]:.4f}")
        history.append({"epoch": epoch, "train_loss": loss, **m})
        if m["map50_95"] > best:
            best = m["map50_95"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best, saved {best_path}")
    pd.DataFrame(history).to_csv(os.path.join(cfg.out_dir, "training_history.csv"), index=False)

    model = load_checkpoint(best_path, NUM_CLASSES + 1, device)
    thresh, rows = sweep_score_threshold(model, val_loader, device, cfg.score_sweep)
    pd.DataFrame(rows).to_csv(os.path.join(cfg.out_dir, "threshold_sweep.csv"), index=False)
    with open(os.path.join(cfg.out_dir, "best_threshold.json"), "w") as f:
        json.dump({"score_thresh": thresh}, f)
    print(f"Best val mAP50-95={best:.4f} | selected score threshold={thresh}")


if __name__ == "__main__":
    main()
