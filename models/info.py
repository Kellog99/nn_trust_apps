from typing import Optional, List, Literal, Any

import timm
from pydantic import AliasChoices, BaseModel, Field, model_validator

from nn_trust import Task

TIMM_MODELS = frozenset(timm.list_models())


class Info(BaseModel):
    """
    This class contains all the information common to the model and dataset `info.json` file
    """
    id: str = Field(
        default=...,
        title="ID",
        description="ID for the file identification"
    )
    name: str = Field(
        default=...,
        title="Name",
        description="File's name, ex. 'Resnet50' or 'Imagenette'"
    )
    date: Optional[str] = Field(
        default=None,
        title="Date",
        description="Date where the file has been generated"
    )
    image: Optional[str] = Field(
        default=None,
        title="Image",
        description="An image that represents the file"
    )
    task: str | Task = Field(
        default=...,
        title="Task",
        description="Task associated with, i.e. classification, detection, etc."
    )
    domain: Optional[str] = Field(
        default=None,
        title="Domain",
        description="Domain where the input belongs"
    )
    num_classes: Optional[int] = Field(
        default=None,
        title="Number of classes",
        description="Number of possible classes."
    )
    file_size: Optional[float] = Field(
        default=None,
        title="Weights",
        description="Size of the file"
    )
    input_dimensionality: List[int] = Field(
        default=...,
        title="Input Dimensionality",
        description="Dimensionality of each input or domain's dimensionality"
    )
    description: Optional[str] = Field(
        default=None,
        title="Description",
        description="Description of the file"
    )
    repository: Optional[str] = Field(
        default=None,
        description="Repository of the dataset/model. It is stored the location where the object is saved."
    )


DATASET_TYPES = Literal[
    "image_folder",
    "flat",
    "parquet",
]


class ParquetInfo(BaseModel):
    image_column: str = Field(
        default="image",
        description="It represents the column of the dataframe where the image is stored."
    )
    image_key: Optional[str] = Field(
        default="bytes",
        validation_alias=AliasChoices("image_key", "label"),
        serialization_alias="image_key",
        description="It represents the key containing encoded image bytes."
    )
    label_column: Optional[str] = Field(
        default="label",
        description="The Parquet column containing the classification target."
    )

    @property
    def label(self) -> Optional[str]:
        """Backward-compatible name for ``image_key`` used by older info files."""
        return self.image_key


class DatasetInfo(Info):
    dataset_type: DATASET_TYPES = Field(
        default="image_folder",
        title="Dataset Format",
        description=(
            "How the repository is loaded: class folders (ImageFolder), a flat "
            "image directory, or automatic detection."
        ),
    )
    folder_data: Optional[str] = Field(
        default=None,
        title="data",
        description="The folder in the dataset folder where the data are stored. Default 'data'",
    )

    num_samples: Optional[int] = Field(
        default=None,
        title="Number of Samples",
        description="Number of samples, i.e. length of the dataset"
    )
    batch_size: int = Field(
        default=32,
        title="Batch Size",
        description="Batch size to use during the Benchmark."
    )
    num_workers: int = Field(
        default=1,
        title="Number of Workers",
        description="Number of workers for handling the dataset's loading."
    )
    label_dict: Optional[dict[int, str]] = Field(
        default=None,
        title="Label Dictionary",
        description="It represent the Label dictionary for extracting the name of the index that the model predicts."
    )
    parquet_info: Optional[ParquetInfo] = Field(
        default=None,
        title="Parquet Information",
        description="It contains all the additional information that are needed for handling the Parquet format dataset."
    )

    @model_validator(mode="after")
    def validate_parquet(self):
        # Older dataset info files identify parquet data solely through this
        # section.  Preserve that format while still allowing an explicit
        # dataset_type to take precedence.
        if "dataset_type" not in self.model_fields_set and self.parquet_info is not None:
            self.dataset_type = "parquet"
        if self.dataset_type == "parquet" and self.parquet_info is None:
            raise ValueError(
                "parquet_info is required when dataset_type is 'parquet'."
            )
        return self


class Transformation(BaseModel):
    mean: list[float]
    std: list[float]
    crop: Optional[int | Any] = None
    size: Optional[int] = None


MODEL_TYPES = Literal[
    "Ollama",
    "Gemini",
    "OpenRouter",
    "HuggingFace",
    "plain",
    "timm",
    "torch_script",
    "torch_dynamo",
    "onnx",
    "api"
]


class ModelInfo(Info):
    dataset: Optional[str] = Field(
        default=None,
        title="Dataset",
        description="Dataset where the model had been optimized on"
    )
    dataset_format: Optional[str] = Field(
        default=None,
        title="Training Dataset Format",
        description=(
            "Format of the data used to train the model, for example ImageFolder, "
            "flat images, COCO, or a custom dataset."
        ),
    )
    parameters: Optional[int] = Field(
        default=None,
        title="Parameters",
        description="Number of the model's parameters"
    )
    transformation: Transformation = Field(
        default=Transformation(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            crop=None,
            size=254,
        ),
        description="It represent the transformation to apply to the input.",
        title="Transformation",
    )
    ################################# Where the model is #################################
    api: Optional[str] = Field(
        default=None,
        title="API",
        description="If the model type is an API then this provide the information to use it."
    )
    model_type: MODEL_TYPES = Field(
        default="plain",
        title="Source Library",
        description="Library where the model has been taken from.",
    )

    ######################################################################################

    @model_validator(mode="after")
    def validate_library_model(self):
        """
        Validates the existence of timm model.
        """
        if self.model_type == "timm":
            if self.id not in TIMM_MODELS:
                raise ValueError(
                    f"You are trying to use a model, {self.id}, from the library {self.model_type} but it doesn't exists."
                )
        elif self.model_type == "HuggingFace" and "/" not in self.id:
            raise ValueError(
                "HuggingFace model should have an id like 'owner/model'"
            )
        elif self.model_type == "Ollama" and "/" in self.id:
            raise ValueError(
                "Ollama model should have an id without '/', e.g. 'llama3:8b-instruct'"
            )

        return self
