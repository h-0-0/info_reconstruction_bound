"""The DP-SGD architectures with GroupNorm (app:exp_spectrum): ResNet-18 for CelebA and ImageNette, WRN-16-4 for
CIFAR-10 (De et al. 2022). BatchNorm couples the examples of a batch and is incompatible with DP-SGD; GroupNorm
normalises each example on its own (Ponomareva et al. 2023, Sec. 5.5.1). LeNet-5 for MNIST is in
:mod:`src.models.lenet`.
"""

from __future__ import annotations

import types

import torch.nn as nn
from opacus.validators import ModuleValidator
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock


# ---------------------------------------------------------------------------
# ResNet-18 (GroupNorm): torchvision's, with BatchNorm converted by Opacus
# ---------------------------------------------------------------------------

def _basicblock_forward_oop(self, x):
    """torchvision BasicBlock.forward with the residual addition out of place."""
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    return self.relu(out)


def _dpfy(model: nn.Module) -> nn.Module:
    """Make a torchvision ResNet compatible with Opacus: BatchNorm -> GroupNorm (ModuleValidator.fix), and no in-place
    operations, which Opacus's per-sample-gradient hooks forbid but the validator does not catch (in-place ReLUs, and
    the ``out += identity`` inside BasicBlock).

    Args:
        model: a torchvision ResNet.

    Returns:
        The converted model (validated).
    """
    model = ModuleValidator.fix(model)
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
        if isinstance(m, BasicBlock):
            m.forward = types.MethodType(_basicblock_forward_oop, m)
    problems = ModuleValidator.validate(model, strict=False)
    assert not problems, f"model still DP-incompatible after fix: {problems}"
    return model


def resnet18_gn(num_classes: int) -> nn.Module:
    """ResNet-18 with GroupNorm, randomly initialised.

    Args:
        num_classes: number of output classes.

    Returns:
        The model.
    """
    return _dpfy(resnet18(weights=None, num_classes=num_classes))


# ---------------------------------------------------------------------------
# WRN-16-4 (GroupNorm): built with GroupNorm and out-of-place operations throughout
# ---------------------------------------------------------------------------

def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(16, c), c)


class WRNBlock(nn.Module):
    """Pre-activation wide-ResNet block: GN-ReLU-conv-GN-ReLU-conv, plus a 1x1 projection shortcut (applied to the
    activated input) when the shape changes, else the identity on the raw input."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.gn1 = _gn(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.gn2 = _gn(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.act = nn.ReLU(inplace=False)
        self.short = (nn.Conv2d(cin, cout, 1, stride, bias=False)
                      if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        o = self.act(self.gn1(x))
        s = self.short(o if not isinstance(self.short, nn.Identity) else x)
        o = self.conv1(o)
        o = self.conv2(self.act(self.gn2(o)))
        return o + s


class WRN(nn.Module):
    """Wide-ResNet-16-4 for 32 x 32 inputs (Zagoruyko & Komodakis): depth 16, two blocks per group, channels
    16 -> 64 -> 128 -> 256."""

    def __init__(self, num_classes: int):
        super().__init__()
        widths = [16, 64, 128, 256]
        self.conv0 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        layers = []
        cin = widths[0]
        for gi, w in enumerate(widths[1:]):
            for b in range(2):
                layers.append(WRNBlock(cin, w, stride=(2 if (gi > 0 and b == 0) else 1)))
                cin = w
        self.blocks = nn.Sequential(*layers)
        self.gn = _gn(cin)
        self.act = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        o = self.conv0(x)
        o = self.act(self.gn(self.blocks(o)))
        return self.fc(self.pool(o).flatten(1))


def wrn16_4_gn(num_classes: int) -> nn.Module:
    """WRN-16-4 with GroupNorm, randomly initialised (and checked by Opacus's validator).

    Args:
        num_classes: number of output classes.

    Returns:
        The model.
    """
    model = WRN(num_classes)
    problems = ModuleValidator.validate(model, strict=False)
    assert not problems, f"WRN-GN DP-incompatible: {problems}"
    return model
