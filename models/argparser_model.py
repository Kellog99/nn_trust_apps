import argparse
from argparse import Namespace
from typing import Optional, Callable, Any, get_origin, get_args, Union, Literal, List

from fastapi import Request
from pydantic import BaseModel, Field


# These are the variables that can be seen by the user, i.e. they are sent to the front end
class SharableVariables(BaseModel):
    host: str = Field(
        default="localhost",
        description="Host to bind the server to (default: localhost)"
    )
    port: int = Field(
        default=8000,
        description="Port to bind the server to (default: 8000)"
    )
    seed: int = Field(
        default=1234,
        description="Seed for the random number generator (default: 1234)"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload (default: False)"
    )

    ########################## PATH ##########################
    path_ds_repo: str = Field(
        default="~/Desktop/StableAI/dataset_repository",
        description="Path to internal storage directory (datasets)"
    )
    path_model_repo: str = Field(
        default="~/Desktop/StableAI/model_repository",
        description="Path to internal storage directory (models)"
    )
    path_model_report_repo: str = Field(
        default="~/Desktop/StableAI/benchmark_repository",
        description="Path to the storage folder for benchmarks and reports."
    )
    ##########################################################

    workers: int = Field(
        default=1,
        description="Number of Uvicorn worker processes"
    )

    ########################## RAY ##########################
    ray_address: Optional[str] = Field(
        default=None,
        description="Ray cluster address (e.g., 127.0.0.1:6379). If None, initializes local cluster"
    )
    ray_py_modules: Optional[str] = Field(
        default=None,
        description="Path to Python modules to include in Ray runtime environment"
    )
    device: str = Field(
        default="cpu",
        description="Device to run the model on"
    )
    #########################################################


class ServerConfig(SharableVariables):
    ########################## FUNCTIONAL ##########################
    function: Literal["app", "benchmark", "report"] = Field(
        default="app",
        description="This represent the function to run."
    )
    ################################################################
    ########################## EXPERIMENT ##########################
    excluded_attacks: list[str] = Field(
        default=[],
        description="List of attacks to exclude"
    )
    excluded_statistics: list[str] = Field(
        default=[],
        description="List of statistics to exclude"
    )
    ################################################################

    path_tmp_files: str = Field(
        default='./tmp',
        description="Path to benchmark output directory"
    )
    max_model_size_upload: int = Field(
        default=5000,
        description="Maximum model file size for upload in MB"
    )
    max_model_json_size_upload: int = Field(
        default=5000,
        description="Maximum model JSON file size for upload in MB"
    )


def config_field(
        attr_name: Optional[str] = None
) -> Callable[[Request], ServerConfig | Any]:
    """
    This function handles the Request typing in the function attributes' typing.
    If the `attr_name` is None than it will return the configuration file
    Otherwise a specific attribute from the configuration file.
    """

    def dependency(request: Request):
        config: ServerConfig = request.app.state.config
        if attr_name:
            return getattr(config, attr_name)
        else:
            return config

    dependency.__name__ = f"get_{attr_name}"
    return dependency


def parsed_argument(model_class: BaseModel) -> Namespace:
    """Add all fields from a Pydantic model to an argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        f"--configuration_file",
        "-cf",
        type=str,
        help="Path to a configuration file."
    )

    for field_name, field_info in model_class.model_fields.items():
        default = field_info.default
        help_text = field_info.description or f"{field_name} parameter"
        required: bool = field_info.is_required()
        field_type: type | None = field_info.annotation

        # Handle Optional/Union types (both typing.Optional and | syntax)
        origin = get_origin(field_type)
        # Check if it's a Union type (including Optional and | syntax)
        if origin is Union:
            args = [a for a in get_args(field_type) if a is not type(None)]
            field_type = args[0] if len(args) == 1 else str
            origin = get_origin(field_type)

        kwargs = {
            "default": default,
            "help": help_text,
            "required": required
        }

        if origin is Literal:
            choices = get_args(field_type)
            kwargs.update(choices=choices, type=type(choices[0]))
        elif field_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs.pop("required", None) if not required else None
        elif origin in (list, List):
            (elem_type,) = get_args(field_type) or (str,)
            kwargs.update(nargs="*", type=elem_type)
        else:
            kwargs["type"] = field_type

        parser.add_argument(f"--{field_name}", **kwargs)

    args = parser.parse_args()
    # Optional: validate through the model itself
    data = {k: v for k, v in vars(args).items() if k != "configuration_file"}
    model_class.model_validate({k: v for k, v in data.items() if v is not None})

    return args
