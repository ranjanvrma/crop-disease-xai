"""A small CNN trained from scratch: the baseline every pretrained model must beat."""

import torch
from torch import nn
from torch.nn import functional as F


def _stage(c_in: int, c_out: int, pool: bool, stride: int = 1) -> nn.Sequential:
    layers = [
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ScratchCNN(nn.Module):
    """Four conv stages -> global average pool -> one linear layer.

    Exposes the same ``forward_features`` / ``forward_head`` /
    ``get_classifier`` interface as timm models, so training, CAM and export
    code treat every architecture alike.
    """

    def __init__(
        self, num_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256), drop_rate: float = 0.2, stem_stride: int = 1
    ):
        super().__init__()
        chans = (3, *widths)
        # stem_stride=2 halves the resolution in the first conv: ~4x fewer FLOPs in every stage.
        self.features = nn.Sequential(*[
            _stage(chans[i], chans[i + 1], pool=i < len(widths) - 1, stride=stem_stride if i == 0 else 1)
            for i in range(len(widths))
        ])
        self.drop_rate = drop_rate
        self.classifier = nn.Linear(widths[-1], num_classes)
        self.num_features = widths[-1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x.mean(dim=(2, 3))
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def get_classifier(self) -> nn.Module:
        return self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))
