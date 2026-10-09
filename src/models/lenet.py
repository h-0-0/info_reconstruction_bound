import torch.nn as nn


class MNISTLeNet5(nn.Module):
    """LeNet-5 for 28x28 MNIST. ~62K params. No normalisation layer at all (Opacus-compatible)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),  # 28 -> 28
            nn.ReLU(),
            nn.AvgPool2d(2),                # 28 -> 14
            nn.Conv2d(6, 16, 5),            # 14 -> 10
            nn.ReLU(),
            nn.AvgPool2d(2),                # 10 -> 5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
