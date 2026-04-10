import json
import tomllib
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "webp", "ppm", "bmp", "pgm", "tif", "tiff"}


def load_metadata(metadata_path: Path) -> dict:
    r"""Loads a metadata file and returns its value as a dictionary.
    Valid metadata file type are 'json', 'toml', 'yaml' and 'yml'.

    :param metadata_path: The path to the metadata file.

    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the file does not have a compatible suffix or it is not a valid TOML, JSON, YAML or YML file.

    :returns: The metadata as a dictionary.
    """
    suffix = metadata_path.suffix.lstrip(".").lower()
    metadata = dict()
    if not metadata_path.exists():
        raise FileNotFoundError(f"The file {metadata_path} could not be found")

    with open(metadata_path, "rb") as f:
        if suffix == "json":
            metadata = json.load(f)
        elif suffix == "toml":
            metadata = tomllib.load(f)
        elif suffix == "yaml" or suffix == "yml":
            metadata = yaml.safe_load(f)
        else:
            raise ValueError(
                f"The file {metadata_path} does not have a compatible suffix. Compatible suffixes are 'json', 'toml', 'yaml', 'yml'"
            )

    return metadata


def has_valid_path_extension(path: Path, extensions: set[str]) -> bool:
    """Checks whether a path has the correct suffix for an image.

    :param path: The path to the image file.
    :type: Path
    :param extensions: The set of valid extensions.
    :type: set[str]

    :returns: True if the path's suffix is contained in the `extensions`, False otherwise.
    """
    return path.suffix.lstrip(".").lower() in extensions


def image_loader_rgb(path: Path) -> Image.Image:
    """Loads an image and convert it to RGB format.

    :param path: The path to the image file.
    :type: Path

    :raises FileNotFoundError: If the file does not exist.
    :raises UnidentifiedImageError: If the file is not a valid image format.
    :raises ValueError: If the image cannot be converted to 'RGB' mode.

    :returns: The image in 'RGB' mode of type :class:`PIL.Image`.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found at {path}")

    try:
        with open(path, "rb") as f:
            img = Image.open(f)

            if img is not None:
                return img.convert("RGB")
            else:
                raise ValueError(f"Cannot convert a 'None' object.\nimg={img}.")
    except Image.UnidentifiedImageError as e:
        raise Image.UnidentifiedImageError(f"File is not a valid image: {path}") from e
    except Exception as e:
        raise ValueError(f"Could not convert image to RGB: {e}") from e
