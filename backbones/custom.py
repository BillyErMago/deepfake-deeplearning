import torch.nn as nn


class CustomBackbone(nn.Module):
    def __init__(self, num_classes, use_stride=False):
        super(CustomBackbone, self).__init__()
        self.use_stride = use_stride

        # Immagini di input: 3x224x224 (Deepfake)

        # Block 1: Primo strato convoluzionale (con downsampling)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2 if use_stride else 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if not use_stride else nn.Identity()
        ) # -> 64x112x112

        # Block 2: Secondo strato convoluzionale (con downsampling)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2 if use_stride else 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if not use_stride else nn.Identity()
        ) # -> 128x56x56

        # Block 3: Terzo strato convoluzionale (con downsampling)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2 if use_stride else 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if not use_stride else nn.Identity()
        ) # -> 256x28x28

        # Block 4: Quarto strato convoluzionale (con downsampling)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2 if use_stride else 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) if not use_stride else nn.Identity()
        ) # -> 256x14x14

        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
