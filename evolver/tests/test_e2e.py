import pytest
import time
from evolver.agent import Agent
from evolver.population import Population
from evolver.evolution import select_parents, create_offspring
from evolver.main import EvolutionaryOptimizer
from evolver.constants import TEST_OPTIMAL_LENGTH

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
    
    # Override running flag to limit iterations
    optimizer.running = True
    
    # Create initial population with random agents
    for _ in range(5):
        agent = Agent(task_chromosome="a" * (_ % 3 + 1))
        agent.score = optimizer.evaluate_agent(agent)
        optimizer.population.add_agent(agent)
        optimizer.statistics.update(agent)
    
    # Run a few iterations
    max_iterations = 10
    iteration = 0
    
    with pytest.raises(StopIteration):  # Just to break out of the loop
        while optimizer.running and iteration < max_iterations:
            iteration += 1
            
            # Select parents
            parents = select_parents(optimizer.population.agents, 4)
            
            # Create parent pairs
            parent_pairs = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    parent_pairs.append((parents[i], parents[i+1]))
            
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
    
    # Check if optimization improved scores
    initial_best = optimizer.population.agents[0].score
    final_best = optimizer.statistics.best_agent.score if optimizer.statistics.best_agent else 0
    
    # The final best score should be at least as good as the initial best
    assert final_best >= initial_best
    
    # Print statistics for debugging
    print(f"Initial best score: {initial_best}")
    print(f"Final best score: {final_best}")
    print(f"Best agent task: {optimizer.statistics.best_agent.chromosomes['task']}")
