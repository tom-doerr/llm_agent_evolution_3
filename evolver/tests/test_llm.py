import pytest
from unittest.mock import MagicMock, patch
from evolver.llm import LLMInterface

@pytest.fixture
def mock_dspy_lm():
    with patch('dspy.LM') as mock_lm:
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.return_value = "Generated text"
        mock_lm.return_value = mock_instance
        yield mock_lm

def test_llm_initialization():
    # Test LLM initialization
    llm = LLMInterface(model_name="test-model")
    assert llm.model_name == "test-model"
    assert llm.lm is None

@pytest.mark.parametrize("prompt,expected", [
    ("Test prompt", "Generated text"),
    ("", "Generated text"),
])
def test_generate_with_mock(mock_dspy_lm, prompt, expected):
    # Test generate method with mock
    llm = LLMInterface(model_name="test-model")
    result = llm.generate(prompt)
    assert result == expected
    
    # Check if LM was initialized
    assert mock_dspy_lm.called

def test_combine_chromosomes_with_llm():
    # Test combine_chromosomes_with_llm method
    llm = LLMInterface()
    
    # Mock the generate method
    llm.generate = MagicMock(return_value="Combined result")
    
    result = llm.combine_chromosomes_with_llm(
        "Parent 1 chromosome",
        "Parent 2 chromosome",
        "Combine these chromosomes"
    )
    
    assert result == "Combined result"
    assert llm.generate.called
    
    # Test with empty chromosomes
    assert llm.combine_chromosomes_with_llm("", "", "") == ""
    assert llm.combine_chromosomes_with_llm("Content", "", "") == "Content"
    assert llm.combine_chromosomes_with_llm("", "Content", "") == "Content"
