import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader

from datasets import HSIDataset
from losses import UnifiedLoss
from model import TGFSNet
from utils import calculate_psnr, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", action="append", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-dir", default="runs/tgfs_net")
    parser.add_argument("--resume")
    parser.add_argument("--rgb-key", default="rgb")
    parser.add_argument("--hsi-key", default="hsi")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--channels", type=int, default=200)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def create_loader(dataset, batch_size, workers, shuffle, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            loss, _ = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item()
        total_psnr += calculate_psnr(outputs.detach().clamp(0, 1), targets).item()
    return total_loss / len(loader), total_psnr / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
            loss, _ = criterion(outputs, targets)
        total_loss += loss.item()
        total_psnr += calculate_psnr(outputs.clamp(0, 1), targets).item()
    return total_loss / len(loader), total_psnr / len(loader)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_sets = [HSIDataset(path, mode="train", patch_size=args.patch_size, rgb_key=args.rgb_key, hsi_key=args.hsi_key) for path in args.train_dir]
    train_set = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_set = HSIDataset(args.val_dir, mode="val", patch_size=args.patch_size, rgb_key=args.rgb_key, hsi_key=args.hsi_key)
    train_loader = create_loader(train_set, args.batch_size, args.workers, True, device)
    val_loader = create_loader(val_set, 1, args.workers, False, device)
    model = TGFSNet(inplanes=3, planes=31, channels=args.channels, num_pyramid_levels=args.levels).to(device)
    criterion = UnifiedLoss().to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    best_psnr = float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_psnr = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_psnr = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        config = vars(args).copy()
        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, scaler, epoch, max(best_psnr, val_psnr), config)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler, scaler, epoch, best_psnr, config)
        print(f"epoch {epoch + 1:03d}/{args.epochs:03d} train_loss {train_loss:.6f} train_psnr {train_psnr:.3f} val_loss {val_loss:.6f} val_psnr {val_psnr:.3f}")


if __name__ == "__main__":
    main()
