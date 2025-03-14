import pytest
from unittest.mock import MagicMock, patch
import dspy
from evolver.dspy_optimizer import DSPyOptimizer

class MockModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = ""
    
    def forward(self, input_text):
        return input_text + self.prompt

@pytest.mark.parametrize("max_evaluations", [10])
def test_dspy_optimizer_initialization():
    # Test basic initialization
    optimizer = DSPyOptimizer(max_agents=100, parallel=5, verbose=True)
    assert optimizer.max_agents == 100
    assert optimizer.parallel == 5
    assert optimizer.verbose == True

@patch('evolver.main.EvolutionaryOptimizer')
def test_dspy_optimizer_optimize(mock_optimizer_class):
    # Create mock objects
    mock_optimizer = MagicMock()
    mock_optimizer_class.return_value = mock_optimizer
    mock_optimizer.statistics.best_agent = MagicMock(chromosomes={"task": "optimized prompt"})
    
    # Create test data
    module = MockModule()
    trainset = ["example1", "example2", "example3"]
    
    # Define simple metric
    def metric(result, example):
        return len(result)
    
    # Create optimizer
    optimizer = DSPyOptimizer(max_agents=100, parallel=5)
    
    # Mock the evaluate_agent method to avoid actual evaluation
    mock_optimizer.evaluate_agent.return_value = 1.0
    
    # Call optimize with limited evaluations
    result = optimizer.optimize(
        module=module,
        metric=metric,
        trainset=trainset,
        max_evaluations=10
    )
    
    # Check if the module was optimized
    assert isinstance(result, MockModule)
    assert result.prompt == "optimized prompt"
    
    # Verify that the optimizer was called correctly
    mock_optimizer_class.assert_called_once()
    assert mock_optimizer.evaluate_agent.called
