import inspect
import annotated_types
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from models import ParametersProps


def _bounds_from_metadata(metadata: list) -> tuple[float | int | None, float | int | None]:
    """Extract (min, max) from annotated_types constraints (Field(ge=, le=, gt=, lt=))."""
    lo = hi = None
    for m in metadata:
        if isinstance(m, annotated_types.Ge):
            lo = m.ge
        elif isinstance(m, annotated_types.Gt):
            lo = m.gt
        elif isinstance(m, annotated_types.Le):
            hi = m.le
        elif isinstance(m, annotated_types.Lt):
            hi = m.lt
    return lo, hi


def extract_numeric_param(name: str, param: inspect.Parameter) -> ParametersProps | None:
    """
    Build a ParametersProps for `param` if it (or its pydantic Field) is int/float.
    Returns None if the parameter is not numeric.
    """
    default_obj = param.default
    annotation = param.annotation

    # Case 1: pydantic Field(...) as default
    if isinstance(default_obj, FieldInfo):
        annotation = default_obj.annotation or annotation
        if annotation not in (int, float):
            return None

        lo, hi = _bounds_from_metadata(default_obj.metadata)

        default = default_obj.default
        if default is PydanticUndefined:
            default = default_obj.default_factory() if default_obj.default_factory else None

        if default is None:
            default = lo if lo is not None else 0

        return ParametersProps(
            id=name,
            name=name,
            min=lo if lo is not None else 0,
            max=hi if hi is not None else 100,
            step=1,
            default=default,
        )

    # Case 2: plain annotation, with or without a literal default
    if annotation not in (int, float):
        return None

    has_default = default_obj is not inspect.Parameter.empty
    default = default_obj if has_default else 0

    return ParametersProps(
        id=name,
        name=name,
        min=0,
        max=100,
        step=1,
        default=default,
    )
