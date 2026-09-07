import torchvision
from PIL import Image as PILImage


class ImageDatasetFolder(torchvision.datasets.ImageFolder):
    def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            is_valid_file=None
    ):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )
        self.data_root = root

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = PILImage.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
