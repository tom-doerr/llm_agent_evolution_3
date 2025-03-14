import pytest
from evolver.agent import Agent
from evolver.evolution import (
    select_parents,
    find_hotspots,
    combine_chromosomes,
    create_offspring
)
from evolver.utils import prepare_weights

def test_prepare_weights():
    # Test with positive scores
    scores = [1.0, 2.0, 3.0, 4.0]
    weights = prepare_weights(scores)
    assert weights == [1.0, 4.0, 9.0, 16.0]
    
    # Test with negative scores
    scores = [-2.0, -1.0, 0.0, 1.0]
    weights = prepare_weights(scores)
    # Should adjust to make all positive
    assert weights[0] < weights[1] < weights[2] < weights[3]

def test_select_parents():
    # Create test agents
    agents = [
        Agent(task_chromosome=f"test{i}", score=float(i))
        for i in range(5)
    ]
    
    # Select parents
    parents = select_parents(agents, 3)
    
    # Check if correct number of parents were selected
    assert len(parents) == 3
    
    # Check if parents are from the original agents
    for parent in parents:
        assert parent in agents

def test_find_hotspots():
    # Test with punctuation
    text = "Hello, world! This is a test."
    hotspots = find_hotspots(text)
    
    # Should include positions of punctuation
    assert 5 in hotspots  # Position of comma
    assert 12 in hotspots  # Position of exclamation mark
    assert 27 in hotspots  # Position of period (index 27)

def test_combine_chromosomes():
    # Create test agents
    parent1 = Agent(task_chromosome="Hello, world!", merging_chromosome="Merge 1")
    parent2 = Agent(task_chromosome="Testing, 123!", merging_chromosome="Merge 2")
    
    # Test combining task chromosomes
    combined = combine_chromosomes(parent1, parent2, "task")
    
    # Result should be a string
    assert isinstance(combined, str)
    
    # Result should not be empty
    assert combined
    
    # Test with empty chromosome
    parent3 = Agent()
    combined = combine_chromosomes(parent1, parent3, "task")
    assert combined == "Hello, world!"

def test_create_offspring():
    # Create test parents
    parent1 = Agent(task_chromosome="Hello, world!", merging_chromosome="Merge 1")
    parent2 = Agent(task_chromosome="Testing, 123!", merging_chromosome="Merge 2")
    
    # Create offspring without LLM
    offspring = create_offspring(parent1, parent2)
    
    # Check if offspring has chromosomes
    assert "task" in offspring.chromosomes
    assert "merging" in offspring.chromosomes
    
    # Chromosomes should not be empty
    assert offspring.chromosomes["task"]
    assert offspring.chromosomes["merging"]
    
    # Test with mock LLM
    from unittest.mock import MagicMock
    mock_llm = MagicMock()
    mock_llm.combine_chromosomes_with_llm.return_value = "LLM combined result"
    
    offspring_with_llm = create_offspring(parent1, parent2, mock_llm)
    
    # Check if LLM was used
    assert mock_llm.combine_chromosomes_with_llm.called
    assert offspring_with_llm.chromosomes["task"] == "LLM combined result"

def test_find_hotspots_debug():
    # Debug test to verify character positions
    text = "Hello, world! This is a test."
    
    # Verify the position of the period
    assert text[27] == "."
