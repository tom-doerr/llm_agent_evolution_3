from unittest.mock import MagicMock
import pytest
from evolver.llm import LLMInterface

def test_llm_initialization():
    # Test initialization
    llm = LLMInterface(model_name="test-model")
    assert llm.model_name == "test-model"
    assert llm.lm is None

@pytest.mark.parametrize("mock_lm", [MagicMock()])
def test_llm_initialize(mock_lm):
    # Test initialize method
    llm = LLMInterface(model_name="test-model")
    llm.initialize()
    
    # Check if LM was created with correct model name
    mock_lm.assert_called_once_with("test-model")
    assert llm.lm is not None

@pytest.mark.parametrize("mock_lm_class", [MagicMock()])
def test_llm_generate(mock_lm_class):
    # Create mock LM
    mock_lm = MagicMock()
    mock_lm_class.return_value = mock_lm
    mock_lm.return_value = "Generated text"
    
    # Test generate method
    llm = LLMInterface(model_name="test-model")
    llm.initialize()
    
    result = llm.generate("Test prompt", max_tokens=10)
    
    # Check if LM was called with correct arguments
    mock_lm.assert_called_once_with("Test prompt", max_tokens=10)
    assert result == "Generated text"

@pytest.mark.parametrize("mock_lm_class", [MagicMock()])
def test_llm_generate_error(mock_lm_class):
    # Create mock LM that raises an exception
    mock_lm = MagicMock()
    mock_lm_class.return_value = mock_lm
    mock_lm.side_effect = Exception("Test error")
    
    # Test generate method with error
    llm = LLMInterface(model_name="test-model")
    llm.initialize()
    
    with pytest.raises(Exception):
        llm.generate("Test prompt")

def test_combine_chromosomes_with_llm():
    # Mock generate method
    mock_generate = MagicMock(return_value="Combined result")
    
    # Test combine_chromosomes_with_llm method
    llm = LLMInterface(model_name="test-model")
    llm.generate = mock_generate
    
    result = llm.combine_chromosomes_with_llm(
        parent1_chromosome="Parent 1",
        parent2_chromosome="Parent 2",
        instruction_chromosome="Combine these",
        max_tokens=10
    )
    
    # Check if generate was called with correct prompt
    assert mock_generate.called
    prompt = mock_generate.call_args[0][0]
    assert "Parent 1" in prompt
    assert "Parent 2" in prompt
    assert "Combine these" in prompt
    
    assert result == "Combined result"

def test_combine_chromosomes_with_llm_error():
    # Mock generate method to raise an exception
    mock_generate = MagicMock(side_effect=Exception("Test error"))
    
    # Test combine_chromosomes_with_llm method with error
    llm = LLMInterface(model_name="test-model")
    llm.generate = mock_generate
    
    result = llm.combine_chromosomes_with_llm(
        parent1_chromosome="Parent 1",
        parent2_chromosome="Parent 2",
        instruction_chromosome="Combine these"
    )
    
    # Check if result contains parts of both parents
    assert "Parent" in result
