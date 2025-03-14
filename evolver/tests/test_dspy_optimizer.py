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

def test_dspy_optimizer_initialization():
    # Test default initialization
    optimizer = DSPyOptimizer()
    assert optimizer.max_agents == 1_000_000
    assert optimizer.parallel == 10
    assert optimizer.verbose == False
    
    # Test custom initialization
    optimizer = DSPyOptimizer(max_agents=100, parallel=5, verbose=True)
    assert optimizer.max_agents == 100
    assert optimizer.parallel == 5
    assert optimizer.verbose == True

@patch('evolver.main.EvolutionaryOptimizer')
def test_dspy_optimizer_optimize(mock_optimizer_class):
    # Create mock objects
    mock_optimizer = MagicMock()
    mock_optimizer_class.return_value = mock_optimizer
    
    # Mock statistics and best agent
    mock_optimizer.statistics.best_agent.chromosomes = {"task": "optimized prompt"}
    
    # Create test data
    module = MockModule()
    metric = lambda result, example: len(result)
    trainset = ["example1", "example2", "example3"]
    
    # Create optimizer
    optimizer = DSPyOptimizer(max_agents=100, parallel=5)
    
    # Run optimization
    with patch('evolver.dspy_optimizer.Agent'):
        optimized_module = optimizer.optimize(
            module=module,
            metric=metric,
            trainset=trainset,
            max_evaluations=10
        )
    
    # Check if optimizer was created with correct args
    mock_optimizer_class.assert_called_once()
    args = mock_optimizer_class.call_args[0][0]
    assert args["max_agents"] == 100
    assert args["parallel"] == 5
    
    # Check if optimized module has the correct prompt
    assert optimized_module.prompt == "optimized prompt"
