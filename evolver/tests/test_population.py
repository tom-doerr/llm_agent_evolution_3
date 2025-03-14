import os
import tempfile
from evolver.agent import Agent
from evolver.population import Population

def test_population_initialization():
    # Test default initialization
    population = Population()
    assert len(population.agents) == 0
    assert population.max_size == 1_000_000

    # Test initialization with custom max_size
    population = Population(max_size=100)
    assert population.max_size == 100

def test_add_remove_agent():
    population = Population(max_size=10)

    # Add agents
    agent1 = Agent(task_chromosome="test1", score=0.5)
    agent2 = Agent(task_chromosome="test2", score=0.7)

    population.add_agent(agent1)
    assert len(population) == 1

    population.add_agent(agent2)
    assert len(population) == 2

    # Remove agent
    population.remove_agent(agent1)
    assert len(population) == 1
    assert population.agents[0] == agent2

def test_get_best_agents():
    population = Population()

    # Add agents with different scores
    agents = [
        Agent(task_chromosome=f"test{i}", score=float(i))
        for i in range(5)
    ]

    for agent in agents:
        population.add_agent(agent)

    # Get best agent
    best = population.get_best_agents(1)
    assert len(best) == 1
    assert best[0].score == 4.0

    # Get top 3 agents
    top3 = population.get_best_agents(3)
    assert len(top3) == 3
    assert [a.score for a in top3] == [4.0, 3.0, 2.0]

def test_population_prune():
    population = Population(max_size=3)

    # Add more agents than max_size
    agents = [
        Agent(task_chromosome=f"test{i}", score=float(i))
        for i in range(5)
    ]

    for agent in agents:
        population.add_agent(agent)

    # Check if population was pruned
    assert len(population) == 3

    # Check if the best agents were kept
    scores = [agent.score for agent in population.agents]
    assert sorted(scores, reverse=True) == [4.0, 3.0, 2.0]

def test_population_save_load():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as tmp:
        filename = tmp.name

    try:
        # Create population with agents
        population1 = Population(max_size=100)
        agents = [
            Agent(task_chromosome=f"test{i}", merging_chromosome=f"merge{i}", score=float(i))
            for i in range(3)
        ]

        for agent in agents:
            population1.add_agent(agent)

        # Save population
        population1.save(filename)

        # Create new population and load from file
        population2 = Population()
        population2.load(filename)

        # Check if loaded population matches original
        assert population2.max_size == population1.max_size
        assert len(population2) == len(population1)

        # Check if agents were loaded correctly
        for i, agent in enumerate(population2.agents):
            assert agent.chromosomes["task"] == f"test{i}"
            assert agent.chromosomes["merging"] == f"merge{i}"
            assert agent.score == float(i)

    finally:
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)
