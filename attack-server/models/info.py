from typing import Optional, List, Literal

import timm
from pydantic import BaseModel, Field, model_validator


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
    task: str = Field(
        default=...,
        title="Task",
        description="Task associated with, i.e. classification, detection, etc."
    )
    domain: str = Field(
        default=...,
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


class DatasetInfo(Info):
    num_sample: Optional[int] = Field(
        default=None,
        title="Number of Samples",
        description="Number of samples, i.e. length of the dataset"
    )
    label_dict: Optional[dict[int, str]] = Field(
        default=None,
        title="Label Dictionary",
        description="It represent the Label dictionary for extracting the name of the index that the model predicts."
    )


class ModelInfo(Info):
    dataset: Optional[str] = Field(
        default=None,
        title="Dataset",
        description="Dataset where the model had been optimized on"
    )
    parameters: Optional[int] = Field(
        default=None,
        title="Parameters",
        description="Number of the model's parameters"
    )
    ################################# Where the model is #################################
    api: Optional[str] = Field(
        default=None,
        title="API",
        description="If the model type is an API then this provide the information to use it."
    )
    model_type: Literal[
        "plain",
        "timm",
        "HuggingFace",
        "torch_script",
        "torch_dynamo",
        "onnx",
        "api"
    ] = Field(
        default="timm",
        title="Library where the model has been taken from.",
    )

    ######################################################################################

    @model_validator(mode="after")
    def validate_library_model(self):
        """
        Validates the existence of timm model.
        """
        if self.model_type in ["timm", "HuggingFace"]:
            lib = timm if self.model_type == "timm" else timm
            if self.id not in lib.list_models():
                raise ValueError(
                    f"You are trying to use a model, {self.id}, from the library {self.model_type} but it doesn't exists."
                )
        return self

