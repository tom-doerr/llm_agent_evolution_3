import dspy
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evolver.dspy_optimizer import DSPyOptimizer
from evolver.examples.dspy_text_classification import run_optimization

# Create a simple DSPy module
class SimpleModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = "Default prompt"

    def forward(self, input_text):
        # Use the prompt to process input
        return f"{input_text} - {self.prompt}"

# Define evaluation metric
def metric(result, example):
    # Simple metric: reward longer outputs
    return len(result)

# Create training data
trainset = ["example1", "example2", "example3"]

def main():
    # Create module
    module = SimpleModule()

    # Run optimization
    optimized_module = run_optimization(
        module=module,
        metric=metric,
        trainset=trainset,
        max_evaluations=20
    )

    # Use optimized module
    result = optimized_module("test")
    print(f"\nFinal result: {result}")

if __name__ == "__main__":
    main()
