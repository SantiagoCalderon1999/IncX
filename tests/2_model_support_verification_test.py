from incx.models.model_enum import ModelEnum


def test_model_support_verification():
    # Given

    # When
    number_of_models = len(ModelEnum)

    # Then
    assert number_of_models >= 3
