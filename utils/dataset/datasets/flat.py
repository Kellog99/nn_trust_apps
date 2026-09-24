from pathlib import Path
from typing import Mapping, Optional

from PIL import Image as PILImage
from torch.utils.data import Dataset


class FlatImageDataset(Dataset):
    """Images in one directory, optionally accompanied by a labels file.

    A labels file can be either ``labels.csv`` (columns ``file`` and ``label``)
    or ``labels.json`` (a mapping from file name to integer label).  Keeping
    this adapter separate from ImageFolder is important: a directory such as
    ``images/`` is not a class called images.
    """

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

    def __init__(self, root: Path, transform=None, labels: Optional[Mapping[str, int]] = None):
        self.root = root
        self.transform = transform
        self.paths = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        self.labels = labels
        if not self.paths:
            raise ValueError(f"No supported images were found in {root}.")
        if labels is not None:
            missing = [path.name for path in self.paths if path.name not in labels]
            if missing:
                raise ValueError(f"The labels file has no label for {missing[0]!r}.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with PILImage.open(self.paths[index]) as image:
            sample = image.convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        # -1 makes an unlabeled image dataset usable for inference while
        # making the absence of labels visible to classification metrics.
        target = -1 if self.labels is None else int(self.labels[self.paths[index].name])
        return sample, target
