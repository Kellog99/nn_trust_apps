from models.info import ModelInfo, Transformation


def _model_info(**kwargs) -> ModelInfo:
    return ModelInfo(
        id="model",
        name="Model",
        task="classification",
        input_dimensionality=[3, 32, 32],
        **kwargs,
    )


def test_transformation_size_is_inferred_from_input_dimensionality():
    model = _model_info()

    assert model.transformation.size == 32


def test_input_dimensionality_takes_precedence_over_transformation_size():
    model = _model_info(
        transformation=Transformation(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            size=224,
        )
    )

    assert model.transformation.size == 32
