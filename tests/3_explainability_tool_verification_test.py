from incrementalexplainer.explainers.explainer_enum import ExplainerEnum

def test_model_support_verification():
    
    # Given
    
    # When
    number_of_explainer = len(ExplainerEnum)
    
    # Then
    assert number_of_explainer > 0
