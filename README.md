# Evolver

An evolutionary algorithm system with LLM-based components.

## Features

- Evolutionary optimization using LLM-based chromosome combination
- Parent selection using Pareto distribution
- Chromosome combination with hotspot-based crossover
- Statistics tracking with sliding window
- Multithreading support
- Save/load functionality with TOML
- External command evaluation support

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
```

## Development

```bash
# Run tests
pytest
```
