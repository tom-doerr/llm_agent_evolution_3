#!/usr/bin/env python3
# Example script to run the evolutionary optimizer on the counting task

import time

# Import evolver modules
from evolver.main import EvolutionaryOptimizer
from evolver.agent import Agent
from evolver.constants import TEST_OPTIMAL_LENGTH

def main():
    # Set up arguments
    args = {
        "parallel": 4,
        "max_agents": 100,
        "verbose": True,
        "eval_command": "python evolver/examples/counting_task.py"
    }
    
    # Create optimizer
    optimizer = EvolutionaryOptimizer(args)
    
    # Initialize with some random agents
    for i in range(5):
        # Create agents with different initial chromosomes
        if i == 0:
            # One with just 'a's
            chromosome = "a" * 5
        elif i == 1:
            # One with mixed characters
            chromosome = "abcdefg"
        elif i == 2:
            # One with spaces
            chromosome = "a a a a a"
        elif i == 3:
            # One with punctuation
            chromosome = "a, a. a! a?"
        else:
            # One with a longer string
            chromosome = "a" * 30
            
        # Create and evaluate agent
        agent = Agent(task_chromosome=chromosome)
        agent.score = optimizer.evaluate_agent(agent)
        optimizer.population.add_agent(agent)
        optimizer.statistics.update(agent)
        
        print(f"Initial agent: {agent}")
    
    # Add a perfect agent to see if evolution can maintain it
    perfect_agent = Agent(task_chromosome="a" * TEST_OPTIMAL_LENGTH)
    perfect_agent.score = optimizer.evaluate_agent(perfect_agent)
    optimizer.population.add_agent(perfect_agent)
    optimizer.statistics.update(perfect_agent)
    print(f"Perfect agent: {perfect_agent}")
    
    # Run for a limited time (30 seconds)
    print("\nStarting evolution...")
    optimizer.running = True
    end_time = time.time() + 30
    
    try:
        while optimizer.running and time.time() < end_time:
            # Run one iteration
            optimizer.run_iteration()
            
            # Print stats every 5 evaluations
            if optimizer.statistics.total_evaluations % 5 == 0:
                optimizer.statistics.print_stats(
                    verbose=optimizer.verbose,
                    population_size=len(optimizer.population)
                )
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        optimizer.running = False
        
    # Print final statistics
    print("\n=== Final Statistics ===")
    optimizer.statistics.print_detailed_stats(population_size=len(optimizer.population))
    
    # Print the best agent's output analysis
    if optimizer.statistics.best_agent:
        best_agent = optimizer.statistics.best_agent
        # In a real implementation, we would generate output from the prompt
        # For now, we use the chromosome as both prompt and output
        prompt = best_agent.chromosomes['task']
        output = prompt  # This would be LLM(prompt) in a real implementation
        
        print(f"\nBest agent score: {best_agent.score}")
        print(f"Best agent prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)
        print(f"Best agent output: {output}")
        print(f"Output length: {len(output)}")
        print(f"Number of 'a's: {output.count('a')}")
        print(f"Optimal length: {TEST_OPTIMAL_LENGTH}")
        print(f"Penalty: {max(0, len(output) - TEST_OPTIMAL_LENGTH)}")

if __name__ == "__main__":
    main()
