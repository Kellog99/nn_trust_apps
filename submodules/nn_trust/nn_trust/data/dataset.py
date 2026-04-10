import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .functional import export_to_image_folder, make_classification_dataset, make_detection_dataset
from .utils import IMAGE_EXTENSIONS, has_valid_path_extension, image_loader_rgb, load_metadata


class LabelLoadingStrategy(ABC):
    r"""Abstract base class for label loading strategies, given a path to an image."""

    def __init__(self, metadata: str | Path | dict | None = None):
        r"""Initializes a :class:`LabelLoadingStrategy` instance.

        :param metadata: a path to a metadata file or an already loaded dictionary. Defaults to ``None``.
        :type metadata: str | Path | dict | None, optional
        """

        if isinstance(metadata, Path | dict):
            self.collect_metadata(metadata)

    def collect_metadata(self, metadata_path: str | Path | dict) -> None:
        r"""Converts an object stored at a path into a meaningful object
        for the chosen :class:`LabelLoadingStrategy`"""
        return None

    @abstractmethod
    def path_to_label(self, path: Path) -> type:
        r"""Maps a given input `path`, to the corresponding `LABEL` type accordingly
        to the chosen :class:`LabelLoadingStrategy`.

        :param path:
        :type path: Path
        """


class ImageFolderLabelLoader(LabelLoadingStrategy):
    r"""A :class:`LabelLoadingStrategy` that loads the label from the parent folder of the image,
    to mimic the behaviour of the :class:`ImageFolder` class.
    """

    def __init__(
            self,
            metadata: str | Path | dict | None = None,
    ):
        super().__init__(metadata)

    def path_to_label(self, path: Path) -> str:
        """Maps a given input `path`, to the corresponding parent folder name.

        :param path: a path object linking to an image file.
        :type path: Path

        :raises ValueError: if the path extension is not valid.
        """
        if has_valid_path_extension(path, IMAGE_EXTENSIONS):
            return str(path.parent.stem)
        else:
            raise ValueError("The path extension is not valid.")


class RegexLabelLoader(LabelLoadingStrategy):
    r"""A :class:`LabelLoadingStrategy` that loads the label from a regex match."""

    def __init__(
            self,
            metadata: str | Path | dict | None = None,
            regex: str = r"^(?:.*[\\/])?([^\\/]+)[\\/][^\\/]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp)$",
            default_label: str = "",
            group: int = 0,
    ):
        r"""Initializes a :class:`RegexLabelLoader`.

        :param regex: a regular expression to match the label, defaults to ``r"^(?:.*[\\/])?([^\\/]+)[\\/][^\\/]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp)$"``.
        :type regex: str, optional
        :param default_label: a default label to be used if no match is found, defaults to ``""``.
        :type default_label: str, optional
        :param group: the group to extract from the regex match, defaults to ``0``.
        :type group: int
        """
        super().__init__(metadata)
        self._regex = regex
        self._group = group
        self._default_label = default_label

    def path_to_label(self, path: Path) -> str:
        """Maps a given input `path`, to the corresponding regex match 0th group.

        :param path: a path object linking to an image file.
        :type path: Path
        """
        matches = re.search(self._regex, str(path))
        if matches:
            return matches.group(self._group)
        else:
            return self._default_label


class MetadataLabelLoader(LabelLoadingStrategy):
    r"""A :class:`LabelLoadingStrategy` that loads the label from a dictionary-like metadata file."""

    def __init__(
            self,
            metadata: str | Path | dict | None,
            path_preprocess: Callable[[Path], Path] | None = None
    ):
        r"""Initializes a :class:`MetadataLabelLoader`.

        :param metadata: a path to a metadata file or an already loaded dictionary.
        :type metadata: str | Path | dict | None
        :param path_preprocess: a function to pre-process the path before loading the metadata. Defaults to ``None``.
        """
        super().__init__(metadata)
        self._path_preprocess = path_preprocess

    def collect_metadata(self, metadata: str | Path | dict) -> None:
        if isinstance(metadata, str):
            metadata = Path(metadata)

        if isinstance(metadata, Path):
            metadata = load_metadata(metadata)
        # NOTE: Here we might add additional transformations depending on something
        self._metadata = metadata

    def path_to_label(self, path: Path) -> Any:
        """Maps a given input `path` to the corresponding metadata value.

        :param path: a path object linking to an image file.
        :type path: Path
        """
        path = path if self._path_preprocess is None else self._path_preprocess(path)
        return self._metadata[path]


class ClassificationDataset(torch.utils.data.Dataset):
    r"""A :class:`torch.utils.data.Dataset` that loads images and labels from a `root` folder.
    It provides a simple and unique class to correctly load diverse image classification datasets,
    with batteries included, see:
        * :meth:`export`
        * :meth:`map_labels`.
        * :meth:`labels`.
        * :meth:`get_label_id`.
        * :meth:`get_label`.

    .. Examples:
        In this example, we load the validation set of Imagenet and convert the labels from the
        wordnet synset IDs to the actual meaning of such IDs.

        >>> import torch
        >>> from nn_trust.data import ClassificationDataset, MetadataLabelLoader
        >>> metadata_label_loader = MetadataLabelLoader(metadata="imagenet1k-labels.json",
        >>>                                             path_preprocess=lambda x: str(x.parent.stem).lower())
        >>> clsdataset = ClassificationDataset(root="./imagenet-1k", label_loader=metadata_label_loader)
        >>> clsdataset.export("./imagenet-1k-image-folder-with-meaningful-labels")

        In this example, we load a less straightforward dataset. Each element's label is stored in the object's path,
        e.g. the path ``"./root/example_fish001.jpeg"`` has label ``"fish"``. Then, we can use a
        :class:`RegexLabelLoader` to correctly load the labels.

        >>> import torch
        >>> from nn_trust.data import ClassificationDataset, RegexLabelLoader
        >>> label_loader = RegexLabelLoader(regex=r"(?<=example_)[a-zA-Z]+(?=\d+)")
        >>> another_dataset = ClassificationDataset(root="./root/", label_loader=label_loader)

        And, we can export it to the same format using :meth:`export`.
    """

    def __init__(
            self,
            root: str | Path,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            label_loader: LabelLoadingStrategy | None = None,
            metadata: str | Path | dict | None = None,
            image_loader: Callable[[Path], Image.Image] | None = None,
            extensions: set[str] | None = None,
    ):
        r"""Initializes a :class:`ClassificationDataset`.

        :param root: The path to the root folder.
        :type root: str | Path
        :param transform: A :class:`Callable` function to apply to the image. Default is ``None``.
        :type transform: Callable
        :param target_transform: A :class:`Callable` function to apply to the label. Default is ``None``.
        :type target_transform: Callable
        :param label_loader: A :class:`LabelLoadingStrategy` object to load the label from the path,
            defaults to :class:`ImageFolderLabelLoader`.
        :type label_loader: LabelLoadingStrategy | None, optional
        :param metadata: The path to the metadata file or an already loaded dictionary, which is then passed to the :class:`LabelLoadingStrategy`,
            defaults to ``None``.
        :type metadata: str | Path | dict | None, optional
        :param image_loader: A :class:`Callable` function to load the image, defaults to :func:`image_loader_rgb`.
        :type image_loader: Callable[[Path], Image.Image] | None, optional
        :param extensions: A set of valid extensions for the images, defaults to `{"jpg", "jpeg", "png", "webp", "ppm", "bmp", "pgm", "tif", "tiff"}`.
        :type extensions: set[str] | None, optional
        """
        super().__init__()
        # convert to Path objects
        if isinstance(root, str):
            root = Path(root)
        if isinstance(metadata, str):
            metadata = Path(metadata)

        # Use as default the same idea for a ImageDataset folder
        if label_loader is None:
            label_loader = ImageFolderLabelLoader(metadata=metadata)

        self._root = root.expanduser()
        self._transform = transform
        self._target_transform = target_transform
        self._extensions = IMAGE_EXTENSIONS if extensions is None else extensions

        # Update metadata if additional metadata was provided
        if metadata is not None:
            label_loader.collect_metadata(metadata)
        self._label_loader = label_loader
        self._loader = image_loader_rgb if image_loader is None else image_loader
        self.make_dataset()

    def make_dataset(self):
        """Generates a list of samples of a form (path_to_sample, class)."""
        if self._extensions is None and not self._extensions:
            self._extensions = IMAGE_EXTENSIONS

        self._instances, self._available_labels = make_classification_dataset(
            self._root,
            path_to_label=self._label_loader.path_to_label,
            path_filter=lambda x: has_valid_path_extension(x, self._extensions),
        )

    def export(self, new_root: str | Path, image_ext: str | None = None) -> None:
        r"""Exports the dataset to an :class:`ImageFolder`-like structure.

        :param new_root: new root folder used to store the dataset.
        :type new_root: `str` or :class:`Path`.
        :param image_ext: new extension to use to store the images. Default is ``None``, which
            keep the extension used in the dataset loading.
        :type image_ext:

        :returns: ``None``.
        """
        if isinstance(new_root, str):
            new_root = Path(new_root)

        if self._instances is None:
            raise RuntimeError(f"The current dataset '{self}', is not correctly initialized.")
        # automatically determines the default suffix
        if image_ext is None:
            # Note: I added a default here, albeit in such cases it has no meaningful application.
            # In case there are no instances, we don't copy data, hence the ".jpeg" extension is not used. - G
            image_ext = self._instances[0][0].suffix if self._instances else ".jpeg"

        export_to_image_folder(self._instances, self._available_labels, new_root, image_ext)

    def map_labels(self, label_map: Callable[[str], str]) -> None:
        r"""Maps the labels of the dataset using a :class:`Callable`.

        :param label_map: A :class:`Callable` that maps a label to a new value.
        :type label_map: Callable

        :returns: ``None``
        """
        og_labels = list(self._available_labels.keys())
        for i in range(len(og_labels)):
            og_label = og_labels[i]
            new_label = label_map(og_label)
            data_idx = self._available_labels.pop(og_label)
            self._available_labels[new_label] = data_idx

    @property
    def labels(self) -> list[str]:
        r"""Returns the list of available labels in the dataset.

        :return: a list of labels.
        :rtype: list[str]
        """
        return list(self._available_labels.keys())

    def get_label(self, idx) -> str:
        r"""Returns the label of a given index.
        :param idx: the index of the sample.
        :type idx: int

        :return: the string label of the sample.
        :rtype: str
        """
        label_id = self.get_label_id(idx)
        labels = self.labels
        return labels[label_id]

    def get_label_id(self, idx) -> int:
        r"""Returns the label id of a given index.

        :param idx: the index of the sample.
        :type idx: int
        """
        return self._instances[idx][1]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        path, target = self._instances[idx]
        sample = self._loader(path)
        if self._transform is not None:
            sample = self._transform(sample)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self._instances)


class DetectionDataset(torch.utils.data.Dataset):
    """The idea is to have a metadata label loader that loads from a dictionary with keys being the path to the image, it gives a torch.Tensor of shape [D, 5]: cls_label, cx, cy, w, h."""

    def __init__(
            self,
            root: str | Path,
            transform: Callable | None = None,
            target_transform: Callable | None = None,
            label_loader: LabelLoadingStrategy | None = None,
            metadata: str | Path | dict | None = None,
            image_loader: Callable[[Path], Image.Image] | None = None,
            extensions: set[str] | None = None,
    ):
        super().__init__()
        # convert to Path objects
        if isinstance(root, str):
            root = Path(root)
        if isinstance(metadata, str):
            metadata = Path(metadata)

        # Use as default the same idea for a ImageDataset folder
        if label_loader is None:
            label_loader = ImageFolderLabelLoader(metadata=metadata)

        self._root = root.expanduser()
        self._transform = transform
        self._target_transform = target_transform
        self._extensions = IMAGE_EXTENSIONS if extensions is None else extensions

        # Update metadata if additional metadata was provided
        if metadata is not None:
            label_loader.collect_metadata(metadata)
        self._label_loader = label_loader
        self._loader = image_loader_rgb if image_loader is None else image_loader
        self.make_dataset()

    def make_dataset(self):
        self._instances = make_detection_dataset(
            self._root,
            self._label_loader.path_to_label,
            path_filter=lambda x: has_valid_path_extension(x, self._extensions),
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self._instances[idx]
        sample = self._loader(path)
        if self._transform is not None:
            sample = self._transform(sample)
        if self._target_transform is not None:
            target = self._target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self._instances)
