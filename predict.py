"""Generate a Kaggle submission CSV from a saved checkpoint.

Usage:
    python predict.py --checkpoint checkpoints/2026-06-05-12-53-42/best_model.pth
    python predict.py --checkpoint checkpoints/.../best_model.pth --threshold 0.5 --save_masks
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eth_mugs_dataset import ETHMugsDataset
from model import UNet, count_parameters
from utils import save_predictions


def predict(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    model = UNet(
        n_channels=3, n_classes=1,
        base_channels=args.base_channels, depth=args.depth,
        bilinear=not args.transposed, dropout=0.0,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device).eval()
    print(f"[INFO] Loaded {args.checkpoint}  ({count_parameters(model):,} params)")

    test_root = os.path.join(args.data_root, "test_data")
    test_ds = ETHMugsDataset(test_root, mode="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    save_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_mask_dir = os.path.join(save_dir, "pred_masks")
    if args.save_masks:
        os.makedirs(out_mask_dir, exist_ok=True)

    image_ids, pred_masks = [], []
    with torch.no_grad():
        for imgs, ids in tqdm(test_loader, desc="predict"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()  # (B,1,H,W)
            preds = (probs[:, 0] > args.threshold).astype(bool)  # (B,H,W)
            for b in range(preds.shape[0]):
                image_ids.append(ids[b])
                pred_masks.append(preds[b])
                if args.save_masks:
                    from PIL import Image
                    Image.fromarray(preds[b]).save(
                        os.path.join(out_mask_dir, f"{ids[b]}_mask.png"))

    csv_path = os.path.join(save_dir, "submission.csv")
    save_predictions(image_ids=image_ids, pred_masks=pred_masks, save_path=csv_path)
    print(f"[INFO] Saved submission to {csv_path}")


def get_args():
    p = argparse.ArgumentParser(description="Generate Kaggle submission from a checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file.")
    p.add_argument("--data_root", default="./datasets")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--transposed", action="store_true", help="Use transposed conv (not bilinear).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_masks", action="store_true", help="Also save predicted masks as PNGs.")
    return p.parse_args()


if __name__ == "__main__":
    predict(get_args())
