import pytest
from evolver.agent import Agent

def test_agent_initialization():
    # Test default initialization
    agent = Agent()
    assert agent.chromosomes["task"] == ""
    assert agent.chromosomes["merging"] == ""
    assert agent.score == 0.0
    assert agent.id is not None
    
    # Test initialization with values
    agent = Agent(task_chromosome="test task", merging_chromosome="test merging", score=0.5)
    assert agent.chromosomes["task"] == "test task"
    assert agent.chromosomes["merging"] == "test merging"
    assert agent.score == 0.5

def test_agent_serialization():
    # Test to_dict and from_dict
    agent1 = Agent(task_chromosome="test task", merging_chromosome="test merging", score=0.5)
    
    # Convert to dict
    agent_dict = agent1.to_dict()
    
    # Create new agent from dict
    agent2 = Agent.from_dict(agent_dict)
    
    # Check if values match
    assert agent2.id == agent1.id
    assert agent2.chromosomes["task"] == agent1.chromosomes["task"]
    assert agent2.chromosomes["merging"] == agent1.chromosomes["merging"]
    assert agent2.score == agent1.score
    assert agent2.creation_timestamp == agent1.creation_timestamp

def test_agent_string_representation():
    # Test __str__ method
    agent = Agent(task_chromosome="test task", merging_chromosome="test merging", score=0.5)
    str_rep = str(agent)
    
    # Check if string contains important information
    assert agent.id in str_rep
    assert "0.5000" in str_rep
    assert "9" in str_rep  # Length of "test task"
