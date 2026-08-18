import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorAwareLoss(nn.Module):
    def rgb_to_lab(self, image):
        image = image.clamp(0, 1)
        if image.shape[1] == 31:
            image = image[:, [20, 15, 10]]
        elif image.shape[1] != 3:
            image = image.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        red, green, blue = image[:, 0], image[:, 1], image[:, 2]

        def linearize(channel):
            return torch.where(channel <= 0.04045, channel / 12.92, ((channel + 0.055) / 1.055).pow(2.4))

        red = linearize(red)
        green = linearize(green)
        blue = linearize(blue)
        x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
        y = 0.2126729 * red + 0.7151522 * green + 0.072175 * blue
        z = (0.0193339 * red + 0.119192 * green + 0.9503041 * blue) / 1.08883

        def transform(value):
            delta = 6.0 / 29.0
            return torch.where(value > delta ** 3, value.clamp_min(0).pow(1.0 / 3.0), value / (3 * delta ** 2) + 4.0 / 29.0)

        fx = transform(x)
        fy = transform(y)
        fz = transform(z)
        return torch.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], dim=1)

    def forward(self, prediction, target):
        prediction = self.rgb_to_lab(prediction)
        target = self.rgb_to_lab(target)
        lightness = F.l1_loss(prediction[:, 0], target[:, 0])
        a = F.l1_loss(prediction[:, 1], target[:, 1])
        b = F.l1_loss(prediction[:, 2], target[:, 2])
        return 0.3 * lightness + 0.7 * torch.sqrt(a.square() + b.square() + 1e-8)


class UnifiedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.25, gamma=0.15, delta=0.1, epsilon=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.color = ColorAwareLoss()

    def gradient_loss(self, prediction, target):
        prediction_x = prediction[..., 1:] - prediction[..., :-1]
        target_x = target[..., 1:] - target[..., :-1]
        prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        return F.l1_loss(prediction_x, target_x) + F.l1_loss(prediction_y, target_y)

    def texture_loss(self, prediction, target):
        prediction_mean = F.avg_pool2d(prediction, 3, stride=1, padding=1)
        target_mean = F.avg_pool2d(target, 3, stride=1, padding=1)
        prediction_var = F.avg_pool2d((prediction - prediction_mean).square(), 3, stride=1, padding=1)
        target_var = F.avg_pool2d((target - target_mean).square(), 3, stride=1, padding=1)
        return F.l1_loss(prediction_var, target_var)

    def frequency_loss(self, prediction, target):
        prediction_spectrum = torch.fft.rfft2(prediction, norm="ortho").abs()
        target_spectrum = torch.fft.rfft2(target, norm="ortho").abs()
        return F.l1_loss(prediction_spectrum, target_spectrum)

    def forward(self, prediction, target):
        l1 = F.l1_loss(prediction, target)
        gradient = self.gradient_loss(prediction, target)
        texture = self.texture_loss(prediction, target)
        frequency = self.frequency_loss(prediction, target)
        color = self.color(prediction, target)
        total = self.alpha * l1 + self.beta * gradient + self.gamma * texture + self.delta * frequency + self.epsilon * color
        return total, {"l1": l1, "gradient": gradient, "texture": texture, "frequency": frequency, "color": color, "total": total}
