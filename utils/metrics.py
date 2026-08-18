import torch


def calculate_psnr(prediction, target, max_value=1.0):
    mse = torch.mean((prediction - target).square())
    return 20 * torch.log10(torch.as_tensor(max_value, device=prediction.device) / torch.sqrt(mse.clamp_min(1e-12)))


def calculate_rmse(prediction, target):
    return torch.sqrt(torch.mean((prediction - target).square()))


def calculate_mrae(prediction, target):
    return torch.mean(torch.abs(prediction - target) / target.abs().clamp_min(1e-6))
