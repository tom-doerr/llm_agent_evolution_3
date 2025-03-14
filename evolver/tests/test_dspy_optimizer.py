from unittest.mock import MagicMock, patch
import dspy
from evolver.dspy_optimizer import DSPyOptimizer

# Create a simple DSPy module for testing
class TestModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = ""
    
    def forward(self, input_text):
        # Simple implementation that just concatenates input with prompt
        return input_text + self.prompt

def test_dspy_optimizer_initialization():
    # Test default initialization
    optimizer = DSPyOptimizer()
    assert optimizer.max_agents == 1_000_000
    assert optimizer.parallel == 10
    assert optimizer.verbose is False
    
    # Test custom initialization
    optimizer = DSPyOptimizer(max_agents=100, parallel=5, verbose=True)
    assert optimizer.max_agents == 100
    assert optimizer.parallel == 5
    assert optimizer.verbose is True

@patch('evolver.main.EvolutionaryOptimizer')
def test_dspy_optimizer_optimize(mock_evolutionary_optimizer):
    # Create mock objects
    mock_optimizer = MagicMock()
    mock_evolutionary_optimizer.return_value = mock_optimizer
    
    # Mock best agent
    mock_best_agent = MagicMock()
    mock_best_agent.chromosomes = {"task": "optimized prompt"}
    mock_optimizer.statistics.best_agent = mock_best_agent
    
    # Create optimizer and test module
    optimizer = DSPyOptimizer(max_agents=100, parallel=5)
    module = TestModule()
    
    # Define simple metric and dataset
    def metric(result, _):
        return len(result)
    
    trainset = ["test1", "test2", "test3"]
    
    # Run optimization
    optimized_module = optimizer.optimize(
        module=module,
        metric=metric,
        trainset=trainset,
        max_evaluations=10
    )
    
    # Check if EvolutionaryOptimizer was created with correct args
    mock_evolutionary_optimizer.assert_called_once()
    args = mock_evolutionary_optimizer.call_args[0][0]
    assert args["max_agents"] == 100
    assert args["parallel"] == 5
    
    # Check if the prompt was updated
    assert optimized_module.prompt == "optimized prompt"
    
    # Check if the optimized module is a TestModule
    assert isinstance(optimized_module, TestModule)
