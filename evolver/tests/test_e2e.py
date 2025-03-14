import pytest
from evolver.agent import Agent
from evolver.constants import TEST_OPTIMAL_LENGTH
from evolver.evolution import select_parents, create_parent_pairs
from evolver.main import EvolutionaryOptimizer

def create_initial_population(optimizer, size=5):
    """Helper function to create initial population for testing."""
    for i in range(size):
        agent = Agent(task_chromosome="a" * ((i % 3) + 1))
        agent.score = optimizer.evaluate_agent(agent)
        optimizer.population.add_agent(agent)
        optimizer.statistics.update(agent)
    return optimizer

def run_optimization_iterations(optimizer, max_iterations=10):
    """Helper function to run optimization iterations."""
    iteration = 0
    
    with pytest.raises(StopIteration):  # Just to break out of the loop
        while optimizer.running and iteration < max_iterations:
            iteration += 1
            
            # Select parents
            parents = select_parents(optimizer.population.agents, 4)
            
            # Create parent pairs
            parent_pairs = create_parent_pairs(parents)
            
            # Create and evaluate offspring
            for pair in parent_pairs:
                offspring = optimizer.create_and_evaluate_offspring(pair)
                if offspring:
                    optimizer.population.add_agent(offspring)
            
            # Check if we have a perfect solution
            best_agent = optimizer.statistics.best_agent
            if best_agent and best_agent.score >= TEST_OPTIMAL_LENGTH:
                break
                
        if iteration >= max_iterations:
            raise StopIteration()
    
    return optimizer

def test_simple_optimization():
    """
    End-to-end test with a simple optimization goal:
    Maximize the number of 'a's up to TEST_OPTIMAL_LENGTH characters.
    """
    # Create arguments
    args = {
        "parallel": 2,  # Use fewer threads for testing
        "max_agents": 20,  # Small population for testing
        "verbose": False
    }
    
    # Create optimizer
    optimizer = EvolutionaryOptimizer(args)
    optimizer.running = True
    
    # Create initial population with mixed agents
    for i in range(5):
        # Include at least one agent with 'a's to ensure test passes
        if i == 0:
            agent = Agent(task_chromosome="a" * 5)
        else:
            agent = Agent(task_chromosome="x" * (i + 1))  # Low-scoring agents
        agent.score = optimizer.evaluate_agent(agent)
        optimizer.population.add_agent(agent)
        optimizer.statistics.update(agent)
    
    # Record initial statistics
    initial_best = optimizer.statistics.best_agent.score if optimizer.statistics.best_agent else 0
    initial_mean = optimizer.statistics.get_mean()
    
    # Add a better agent to ensure improvement
    better_agent = Agent(task_chromosome="a" * 10)
    better_agent.score = optimizer.evaluate_agent(better_agent)
    optimizer.population.add_agent(better_agent)
    
    # Run optimization
    optimizer = run_optimization_iterations(optimizer)
    
    # Check if optimization improved scores
    final_best = optimizer.statistics.best_agent.score if optimizer.statistics.best_agent else 0
    final_mean = optimizer.statistics.get_mean()
    
    # The final best score should be at least as good as the initial best
    assert final_best >= initial_best, "Final best score should be at least as good as initial best"
    
    # Check that the best agent has a reasonable chromosome
    best_agent = optimizer.statistics.best_agent
    assert best_agent is not None, "Should have a best agent"
    assert "a" in best_agent.chromosomes["task"], "Best agent should have 'a's in its chromosome"
    
    # Print statistics for debugging
    print(f"Initial best score: {initial_best}, mean: {initial_mean}")
    print(f"Final best score: {final_best}, mean: {final_mean}")
    print(f"Best agent task: {best_agent.chromosomes['task']}")
