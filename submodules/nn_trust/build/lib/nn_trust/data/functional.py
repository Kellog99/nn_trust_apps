import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

def export_to_image_folder(
        instances: list[tuple[Path, int]],
        available_labels: dict[str, list[int]],
        new_root: Path,
        image_ext: str,
) -> None:
    """Exports the dataset to an :class:`ImageFolder`-like structure.

    :param instances: list of tuples containing the path to the image and its corresponding label ID.
    :type instances: list[tuple[Path, int]].
    :param available_labels: dictionary containing the labels and their corresponding indices in the instances list.
    :type available_labels: dict[str, list[int]].
    :param new_root: new root folder used to store the dataset.
    :type new_root: str or Path.
    :param image_ext: the image extension the dataset images should be stored with.
    :type image_ext: str.

    :raises FileNotFoundError: If a source file is not found.

    :returns: ``None``.
    """
    # Create new_root if it does not exists
    new_root.mkdir(parents=True, exist_ok=True)

    # Iterate throughout the dataset to store the values
    for cls_name in available_labels:
        cls_name_path = new_root / cls_name
        cls_name_path.mkdir(exist_ok=True)
        for idx in available_labels[cls_name]:
            og_path, _ = instances[idx]
            new_path = cls_name_path / og_path.name
            new_path = new_path.with_suffix(image_ext)
            # In case there is a change of image extension, we need to convert it using PIL.
            if og_path.suffix != image_ext:
                with Image.open(og_path) as img:
                    img.save(new_path)
            # Otherwise copy the data using shutil which is faster.
            else:
                try:
                    # copyfile for copying files quickly
                    shutil.copyfile(og_path, new_path)
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Error: The source file {og_path} was not found.") from e


def make_classification_dataset(
        root: Path,
        path_to_label: Callable[[Path], str],
        path_filter: Callable[[Path], bool] | None = None,
) -> tuple[list[tuple[Path, int]], dict[str, list[int]]]:
    r"""Indexes a root path associating to each data element a corresponding label accordingly to
    a specified `path_to_label` function.

    :param root: root :class:`Path` object that specifies the entry point of the dataset.
    :type root: str or Path.
    :param path_to_label: function that maps a :class:`Path` object to a unique string corresponding to the data point
        label.
    :type path_to_label: Callable[[Path], bool].
    :param path_filter: function that determines whether a :class:`Path` should be included in the dataset or not.
    :type path_filter: Callable[[Path], bool] or None.

    :returns: a tuple containing the instances list and the available labels dictionary. The labels dictionary
        is a dictionary containing the labels and their corresponding indices in the instances list.
    :rtype: tuple[list[tuple[Path, int]], dict[str, list[int]]].
    """
    if not root.exists():
        raise ValueError(f"The given path {root} does not exist.")

    # collect all instances as a list of tuple (Path, int), where `int` is the number of the class
    instances: list[tuple[Path, int]] = []
    available_labels = dict()
    idx = 0
    # Collect and iterate with order preserving, for correct numbering
    for available_path in sorted(root.rglob("*")):
        # Check if the file is available and non-empty, else skip
        if not available_path.is_file() or available_path.stat().st_size == 0:
            continue
        # Add the label if the path satisfies the path_filter criterion
        path_check = path_filter(available_path) if path_filter is not None else True
        if path_check:
            class_label = path_to_label(available_path)

            # Add to each class label the idx of the corresponding data element
            if class_label in available_labels:
                available_labels[class_label].append(idx)
            else:
                available_labels[class_label] = [idx]

            # Prepare the item to be appended to the instances list
            item = available_path, list(available_labels.keys()).index(class_label)
            instances.append(item)
            idx += 1
    return instances, available_labels


def make_detection_dataset(
        root: Path,
        path_to_label: Callable[[Path], list[torch.Tensor]],
        path_filter: Callable[[Path], bool] | None = None,
) -> tuple[list[tuple[Path, list[int]]], dict[str, list[int]]]:
    if not root.exists():
        raise ValueError(f"The given path {root} does not exist.")

    # collect all instances as a list of tuple (Path, int), where `int` is the number of the class
    instances: list[tuple[Path, list[torch.Tensor]]] = []
    # Collect and iterate with order preserving, for correct numbering
    for available_path in sorted(root.rglob("*")):
        # Check if the file is available and non-empty, else skip
        if not available_path.is_file() or available_path.stat().st_size == 0:
            continue
        # Add the label if the path satisfies the path_filter criterion
        path_check = path_filter(available_path) if path_filter is not None else True
        if path_check:
            detections = path_to_label(available_path)
            item = available_path, detections
            instances.append(item)
    return instances