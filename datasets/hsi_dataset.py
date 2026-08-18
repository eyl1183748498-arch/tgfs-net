from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class HSIDataset(Dataset):
    def __init__(self, root, mode="train", patch_size=64, rgb_key="rgb", hsi_key="hsi"):
        self.root = Path(root)
        self.mode = mode
        self.patch_size = patch_size
        self.rgb_key = rgb_key
        self.hsi_key = hsi_key
        self.files = sorted(self.root.rglob("*.mat"))
        if not self.files:
            raise FileNotFoundError(f"No .mat files found in {self.root}")

    def __len__(self):
        return len(self.files)

    def _to_chw(self, array, channels):
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D array, got {tuple(tensor.shape)}")
        if tensor.shape[0] == channels:
            return tensor
        if tensor.shape[-1] == channels:
            return tensor.permute(2, 0, 1)
        if tensor.shape[1] == channels:
            return tensor.permute(1, 0, 2)
        axis = min(range(3), key=lambda i: tensor.shape[i])
        return tensor.movedim(axis, 0)

    def _normalize(self, rgb, hsi):
        rgb = torch.nan_to_num(rgb)
        hsi = torch.nan_to_num(hsi)
        if rgb.max() > 1:
            rgb = rgb / 255.0
        if hsi.max() > 1:
            hsi = hsi / hsi.max().clamp_min(1e-8)
        return rgb.clamp(0, 1), hsi.clamp(0, 1)

    def _crop(self, rgb, hsi):
        if self.patch_size is None or self.patch_size <= 0:
            return rgb, hsi
        size = self.patch_size
        height, width = rgb.shape[-2:]
        if height < size or width < size:
            target_height = max(height, size)
            target_width = max(width, size)
            rgb = F.interpolate(rgb.unsqueeze(0), size=(target_height, target_width), mode="bilinear", align_corners=False).squeeze(0)
            hsi = F.interpolate(hsi.unsqueeze(0), size=(target_height, target_width), mode="bilinear", align_corners=False).squeeze(0)
            height, width = target_height, target_width
        if self.mode == "train":
            top = torch.randint(0, height - size + 1, (1,)).item()
            left = torch.randint(0, width - size + 1, (1,)).item()
        else:
            top = (height - size) // 2
            left = (width - size) // 2
        return rgb[:, top:top + size, left:left + size], hsi[:, top:top + size, left:left + size]

    def _augment(self, rgb, hsi):
        if self.mode != "train":
            return rgb, hsi
        if torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-1,))
            hsi = torch.flip(hsi, dims=(-1,))
        if torch.rand(()) < 0.5:
            rgb = torch.flip(rgb, dims=(-2,))
            hsi = torch.flip(hsi, dims=(-2,))
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            rgb = torch.rot90(rgb, k, dims=(-2, -1))
            hsi = torch.rot90(hsi, k, dims=(-2, -1))
        return rgb, hsi

    def __getitem__(self, index):
        path = self.files[index]
        with h5py.File(path, "r") as file:
            if self.rgb_key not in file or self.hsi_key not in file:
                raise KeyError(f"Missing {self.rgb_key} or {self.hsi_key} in {path}")
            rgb = self._to_chw(file[self.rgb_key][()], 3)
            hsi = self._to_chw(file[self.hsi_key][()], 31)
        if rgb.shape[-2:] != hsi.shape[-2:]:
            hsi = F.interpolate(hsi.unsqueeze(0), size=rgb.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
        rgb, hsi = self._normalize(rgb, hsi)
        rgb, hsi = self._crop(rgb, hsi)
        rgb, hsi = self._augment(rgb, hsi)
        return {"input": rgb.contiguous(), "target": hsi.contiguous(), "name": str(path.relative_to(self.root))}
