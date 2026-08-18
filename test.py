import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import HSIDataset
from model import TGFSNet
from utils import calculate_mrae, calculate_psnr, calculate_rmse, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-dir")
    parser.add_argument("--rgb-key", default="rgb")
    parser.add_argument("--hsi-key", default="hsi")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--channels", type=int, default=200)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    dataset = HSIDataset(args.test_dir, mode="test", patch_size=args.patch_size, rgb_key=args.rgb_key, hsi_key=args.hsi_key)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    model = TGFSNet(inplanes=3, planes=31, channels=args.channels, num_pyramid_levels=args.levels).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    totals = {"psnr": 0.0, "rmse": 0.0, "mrae": 0.0}
    count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        outputs = model(inputs)
        if outputs.shape[-2:] != targets.shape[-2:]:
            outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        outputs = outputs.clamp(0, 1)
        batch_size = inputs.shape[0]
        for index in range(batch_size):
            prediction = outputs[index:index + 1]
            target = targets[index:index + 1]
            totals["psnr"] += calculate_psnr(prediction, target).item()
            totals["rmse"] += calculate_rmse(prediction, target).item()
            totals["mrae"] += calculate_mrae(prediction, target).item()
            if save_dir:
                name = Path(batch["name"][index]).stem
                np.save(save_dir / f"{name}.npy", prediction.squeeze(0).cpu().numpy())
            count += 1
    print(f"PSNR {totals['psnr'] / count:.4f}")
    print(f"RMSE {totals['rmse'] / count:.6f}")
    print(f"MRAE {totals['mrae'] / count:.6f}")


if __name__ == "__main__":
    main()
