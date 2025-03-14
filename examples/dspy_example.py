import dspy
from evolver.dspy_optimizer import DSPyOptimizer

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
    # Create optimizer
    optimizer = DSPyOptimizer(
        max_agents=20,  # Small population for demo
        parallel=2,     # Few parallel agents for demo
        verbose=True
    )

    # Create module
    module = SimpleModule()

    print("Starting optimization...")

    # Optimize module
    optimized_module = optimizer.optimize(
        module=module,
        metric=metric,
        trainset=trainset,
        max_evaluations=20  # Few evaluations for demo
    )

    # Use optimized module
    result = optimized_module("test")
    print(f"\nFinal result: {result}")
    print(f"Optimized prompt: {optimized_module.prompt}")

if __name__ == "__main__":
    main()
