import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm


def _to_device(imgs, targets, device):
    imgs = [i.to(device) for i in imgs]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return imgs, targets


def train_one_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
    for imgs, targets, _ in pbar:
        imgs, targets = _to_device(imgs, targets, device)
        loss = sum(model(imgs, targets).values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        pbar.set_postfix(loss=float(loss.item()))
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_map(model, loader, device, score_thresh: float = 0.0):
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for imgs, targets, _ in loader:
        preds = model([i.to(device) for i in imgs])
        out = []
        for p in preds:
            keep = p["scores"] >= score_thresh
            out.append({k: v[keep].cpu() for k, v in p.items()})
        metric.update(out, [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets])
    res = metric.compute()
    return {"map50_95": float(res["map"]), "map50": float(res["map_50"])}


def sweep_score_threshold(model, loader, device, candidates):
    """Return (best_threshold_by_mAP50, list_of_rows)."""
    rows = []
    for sc in candidates:
        m = evaluate_map(model, loader, device, score_thresh=sc)
        rows.append({"score_thresh": sc, **m})
    best = max(rows, key=lambda r: r["map50"])["score_thresh"]
    return best, rows
