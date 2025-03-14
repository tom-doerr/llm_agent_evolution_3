# Evolver

An evolutionary algorithm system with LLM-based components.

## Features

- Evolutionary optimization using LLM-based chromosome combination
- Parent selection using Pareto distribution
- Chromosome combination with hotspot-based crossover
- Statistics tracking with sliding window
- Multithreading support with thread safety
- Save/load functionality with TOML
- External command evaluation support
- Rich tables for detailed statistics
- DSPy optimizer interface

## Installation

```bash
# Install with poetry
poetry install
```

## Usage

```bash
# Run with default settings
python -m evolver.main

# Run with custom settings
python -m evolver.main --parallel 20 --verbose

# Run with external evaluation command
python -m evolver.main --eval-command "python evaluate.py"

# Save and load population
python -m evolver.main --save population.toml
python -m evolver.main --load population.toml

# Pipe input to the agent
echo "initial chromosome" | python -m evolver.main
```

### Running the Counting Task Example

The repository includes an example "counting task" that rewards agents for maximizing the number of 'a' characters up to a certain length, with penalties for exceeding that length:

```bash
# Run the counting task example
python evolver/examples/run_counting_task.py

# Or run directly with the main module
python -m evolver.main --eval-command "python evolver/examples/counting_task.py" --verbose
```

## Using as a DSPy Optimizer

```python
import dspy
from evolver.dspy_optimizer import DSPyOptimizer

# Create a DSPy module
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = ""
    
    def forward(self, input_text):
        # Use the prompt
        return input_text + self.prompt

# Create optimizer
optimizer = DSPyOptimizer(
    max_agents=1000,
    parallel=10,
    verbose=True
)

# Define evaluation metric
def metric(result, example):
    # Calculate score based on your requirements
    # Higher score = better result
    return len(result)

# Create training data
trainset = ["example1", "example2", "example3"]

# Optimize module
module = MyModule()
optimized_module = optimizer.optimize(
    module=module,
    metric=metric,
    trainset=trainset,
    max_evaluations=100
)

# Use optimized module
result = optimized_module("test")
print(f"Result: {result}")
print(f"Optimized prompt: {optimized_module.prompt}")
```

## Development

```bash
# Run tests
pytest
```
