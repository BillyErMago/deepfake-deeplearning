import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from backbones.custom import CustomBackbone


class Net(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super(Net, self).__init__()
        if backbone_name == "custom":
            self.backbone = CustomBackbone(num_classes=num_classes, use_stride=False)
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            
            # Unfreeze all layers for fine-tuning on deepfake task
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(
                f"Backbone '{backbone_name}' non supportata. Usa 'resnet50' o 'custom'."
            )

    def forward(self, x):
        return self.backbone(x)
