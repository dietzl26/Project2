"""Train a from-scratch U-Net on the ETHMugs dataset.

Examples
--------
Main model (strong default):
    python train.py --data_root ./datasets --epochs 120 --base_channels 64 --depth 4

Two configurations required by the project (run both, compare val IoU):
    python train.py --base_channels 32 --depth 4          # narrow
    python train.py --base_channels 64 --depth 4          # wide
  or depth instead of width:
    python train.py --base_channels 48 --depth 3          # shallow
    python train.py --base_channels 48 --depth 4          # deep

Baseline comparison (the simplistic lecture U-Net):
    python train.py --arch baseline --epochs 120

After training it writes:
    <ckpt_dir>/<timestamp>/best_model.pth   (best val-IoU checkpoint)
    <ckpt_dir>/<timestamp>/last_epoch.pth
    <ckpt_dir>/<timestamp>/submission.csv   (Kaggle file, if --predict)
"""

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eth_mugs_dataset import ETHMugsDataset
from utils import compute_iou, save_predictions
from model import UNet, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiceLoss(nn.Module):
    """Soft Dice loss on the sigmoid of the logits."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        target = target.reshape(target.size(0), -1)
        inter = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return (1 - dice).mean()


class ComboLoss(nn.Module):
    """BCE-with-logits + Dice. Works well for imbalanced segmentation."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        return self.bce_weight * self.bce(logits, target) + (1 - self.bce_weight) * self.dice(logits, target)


def build_model(args):
    if args.arch == "baseline":
        # The simplistic lecture U-Net, used as the comparison approach.
        from unet_simplistic import UNet as SimplisticUNet
        model = SimplisticUNet(3, 1)
        print("[INFO] Using baseline (simplistic) U-Net.")
    else:
        model = UNet(
            n_channels=3, n_classes=1,
            base_channels=args.base_channels, depth=args.depth,
            bilinear=args.bilinear, dropout=args.dropout,
        )
        print(f"[INFO] Using U-Net | base_channels={args.base_channels} "
              f"depth={args.depth} bilinear={args.bilinear}")
    print(f"[INFO] Trainable parameters: {count_parameters(model):,}")
    return model


@torch.no_grad()
def evaluate(model, loader, device, thresholds=(0.5,)):
    """Return a dict {threshold: mean IoU} over the loader."""
    model.eval()
    sums = {t: 0.0 for t in thresholds}
    n = 0
    for image, gt_mask in loader:
        image = image.to(device)
        probs = torch.sigmoid(model(image)).cpu().numpy()  # (B,1,H,W)
        gt = gt_mask.numpy().astype(int)
        for b in range(probs.shape[0]):
            g = gt[b, 0]
            for t in thresholds:
                p = (probs[b, 0] > t).astype(int)
                sums[t] += compute_iou(p, g)
            n += 1
    return {t: sums[t] / max(n, 1) for t in thresholds}


def predict_and_save(model, test_loader, device, threshold, save_dir, save_masks=False):
    model.eval()
    image_ids, pred_masks = [], []
    out_mask_dir = os.path.join(save_dir, "pred_masks")
    if save_masks:
        os.makedirs(out_mask_dir, exist_ok=True)
    with torch.no_grad():
        for imgs, ids in tqdm(test_loader, desc="predict"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()  # (B,1,H,W)
            preds = (probs[:, 0] > threshold).astype(bool)    # (B,H,W)
            for b in range(preds.shape[0]):
                image_ids.append(ids[b])
                pred_masks.append(preds[b])
                if save_masks:
                    from PIL import Image
                    Image.fromarray(preds[b]).save(
                        os.path.join(out_mask_dir, f"{ids[b]}_mask.png"))
    csv_path = os.path.join(save_dir, "submission.csv")
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=csv_path)
    print(f"[INFO] Wrote submission to {csv_path}")


def train(args, ckpt_dir):
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():   # Apple Silicon / Metal GPU
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    train_root = os.path.join(args.data_root, "train_data")
    test_root = os.path.join(args.data_root, "test_data")

    # ---- Train / validation split (by image id, fixed seed) ----
    all_ids = sorted(
        f.replace("_rgb.jpg", "")
        for f in os.listdir(os.path.join(train_root, "rgb"))
        if f.endswith("_rgb.jpg")
    )
    rng = random.Random(args.seed)
    rng.shuffle(all_ids)
    n_val = max(1, int(round(len(all_ids) * args.val_frac)))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"[INFO] {len(train_ids)} train / {len(val_ids)} val images")

    train_ds = ETHMugsDataset(train_root, mode="train", augment=not args.no_augment, image_ids=train_ids)
    val_ds = ETHMugsDataset(train_root, mode="val", augment=False, image_ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args).to(device)
    criterion = ComboLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_iou = -1.0
    best_path = os.path.join(ckpt_dir, "best_model.pth")

    print("[INFO] Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for image, gt_mask in pbar:
            image = image.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(image)
                loss = criterion(logits, gt_mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * image.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_loss = running / len(train_ds)

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

        if (epoch + 1) % args.val_frequency == 0 or epoch == args.epochs - 1:
            iou = evaluate(model, val_loader, device, thresholds=(0.5,))[0.5]
            print(f"[INFO] epoch {epoch + 1}: train_loss={train_loss:.4f}  val_IoU(@0.5)={iou * 100:.2f}")
            if iou > best_iou:
                best_iou = iou
                torch.save(model.state_dict(), best_path)
                print(f"[INFO]   -> new best ({best_iou * 100:.2f}), saved {best_path}")

    print(f"[INFO] Best val IoU @0.5: {best_iou * 100:.2f}")

    # ---- Threshold tuning on validation (best checkpoint) ----
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    thresholds = [round(0.30 + 0.05 * k, 2) for k in range(9)]  # 0.30 .. 0.70
    results = evaluate(model, val_loader, device, thresholds=thresholds)
    best_t = max(results, key=results.get)
    print("[INFO] Validation IoU per threshold:")
    for t in thresholds:
        marker = "  <-- best" if t == best_t else ""
        print(f"         thr={t:.2f}  IoU={results[t] * 100:.2f}{marker}")

    # ---- Predict on the test set ----
    if args.predict:
        test_ds = ETHMugsDataset(test_root, mode="test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)
        predict_and_save(model, test_loader, device, threshold=best_t,
                         save_dir=ckpt_dir, save_masks=args.save_masks)


def get_args():
    p = argparse.ArgumentParser(description="SML Project 2 - ETH Mugs segmentation.")
    p.add_argument("-d", "--data_root", default="./datasets",
                   help="Folder containing train_data/ and test_data/.")
    p.add_argument("--ckpt_dir", default="./checkpoints")
    # model
    p.add_argument("--arch", default="unet", choices=["unet", "baseline"])
    p.add_argument("--base_channels", type=int, default=64, help="Width (channels at level 0).")
    p.add_argument("--depth", type=int, default=4, help="Number of down/up levels.")
    p.add_argument("--bilinear", action="store_true", default=True)
    p.add_argument("--transposed", dest="bilinear", action="store_false",
                   help="Use transposed conv upsampling instead of bilinear.")
    p.add_argument("--dropout", type=float, default=0.0)
    # optimization
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--bce_weight", type=float, default=0.5)
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision.")
    # data
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_frequency", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # output
    p.add_argument("--predict", action="store_true", default=True,
                   help="Generate submission.csv on the test set after training.")
    p.add_argument("--no_predict", dest="predict", action="store_false")
    p.add_argument("--save_masks", action="store_true",
                   help="Also save predicted masks as PNGs (for visual inspection).")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO] Checkpoints/outputs will be saved to:", ckpt_dir)
    train(args, ckpt_dir)
