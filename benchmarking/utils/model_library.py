from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class ResNet50Dirichlet(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 prior: float = 1):
        super(ResNet50Dirichlet, self).__init__()

        self.backbone = resnet50()
        self.prior = prior

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return F.softplus(self.backbone(x)) + self.prior
    
models_library = {
    "resnet50dirichlet": ResNet50Dirichlet
}