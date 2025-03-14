from typing import List, Tuple, Callable, Any, Optional
import dspy

from .agent import Agent
from .main import EvolutionaryOptimizer
from .utils import create_parent_pairs

class DSPyOptimizer:
    """
    DSPy optimizer interface for the evolutionary algorithm.
    
    This class provides an interface to use the evolutionary algorithm
    as an optimizer for DSPy pipelines.
    """
    
    def __init__(self, 
                 max_agents: int = 1_000_000,
                 parallel: int = 10,
                 verbose: bool = False):
        """
        Initialize the DSPy optimizer.
        
        Args:
            max_agents: Maximum number of agents in the population
            parallel: Number of agents to run in parallel
            verbose: Enable verbose output
        """
        self.max_agents = max_agents
        self.parallel = parallel
        self.verbose = verbose
    
    
    def optimize(self, 
                 module: dspy.Module,
                 metric: Callable,
                 trainset: List[Any],
                 valset: Optional[List[Any]] = None,
                 max_evaluations: int = 1000,
                 **kwargs) -> dspy.Module:
        """
        Optimize a DSPy module using evolutionary algorithm.
        
        Args:
            module: DSPy module to optimize
            metric: Function to evaluate module performance
            trainset: Training dataset
            valset: Validation dataset (optional)
            max_evaluations: Maximum number of evaluations
            **kwargs: Additional arguments
            
        Returns:
            Optimized DSPy module
        """
        # Create a copy of the module to optimize
        optimized_module = module.__class__()
        
        # Set up the evolutionary optimizer
        args = {
            "parallel": self.parallel,
            "max_agents": self.max_agents,
            "verbose": self.verbose,
            "model": kwargs.get("model", "openrouter/google/gemini-2.0-flash-001")
        }
        
        optimizer = EvolutionaryOptimizer(args)
        
        # Set up the evaluation function
        optimizer.evaluate_agent = self._create_evaluation_function(
            optimized_module, metric, trainset
        )
        
        # Initialize population with a few random agents
        # Start with empty prompts and let evolution discover good ones
        for i in range(10):
            # Create initial prompts with minimal guidance
            initial_prompt = ""
            if i > 0:  # Add some minimal variation after the first agent
                initial_prompt = f"Input: {{input}}"
                
            agent = Agent(task_chromosome=initial_prompt)
            agent.score = optimizer.evaluate_agent(agent)
            optimizer.population.add_agent(agent)
            optimizer.statistics.update(agent)
        
        # Run optimization for a limited number of evaluations
        optimizer.running = True
        evaluation_count = 0
        
        while optimizer.running and evaluation_count < max_evaluations:
            # Select parents
            num_pairs = max(1, min(10, self.parallel))  # Ensure it's a reasonable integer
            num_parents = num_pairs * 2
            
            # Get candidates for parents
            if len(optimizer.population.agents) > 0:
                parents = optimizer.population.get_candidates(num_parents)
            else:
                # Handle empty population case
                break
            
            # Create parent pairs
            parent_pairs = create_parent_pairs(parents)
            
            # Create and evaluate offspring
            for pair in parent_pairs:
                offspring = optimizer.create_and_evaluate_offspring(pair)
                if offspring:
                    optimizer.population.add_agent(offspring)
                    evaluation_count += 1
                
                if evaluation_count >= max_evaluations:
                    break
            
            # Print statistics periodically
            if evaluation_count % 10 == 0:
                optimizer.statistics.print_stats(
                    verbose=optimizer.verbose, 
                    population_size=len(optimizer.population)
                )
        
        # Get the best agent and apply its chromosome to the module
        best_agent = optimizer.statistics.best_agent
        if best_agent:
            optimized_module.prompt = best_agent.chromosomes["task"]
            
            # Print final best prompt if verbose
            if optimizer.verbose:
                print(f"\nOptimized prompt:\n{optimized_module.prompt}\n")
        
        return optimized_module
        
    def _create_evaluation_function(self, module, metric, trainset):
        """Create an evaluation function for the DSPy module."""
        def evaluate_module(agent: Agent) -> float:
            """Evaluate a DSPy module using the agent's task chromosome as prompt."""
            # Apply the agent's task chromosome as the module's prompt
            module.prompt = agent.chromosomes["task"]
            
            # Evaluate on training set
            total_score = 0.0
            num_examples = min(len(trainset), 10)  # Limit to 10 examples for efficiency
            successful_examples = 0
            
            for example in trainset[:num_examples]:
                try:
                    result = module(example)
                    score = metric(result, example)
                    total_score += score
                    successful_examples += 1
                except (ValueError, TypeError) as error:
                    print(f"Error evaluating module: {error}")
                    # Continue with other examples instead of returning 0
                except Exception as error:
                    print(f"Unexpected error evaluating module: {error}")
                    # Continue with other examples
            
            # Return average score based on successful examples
            if successful_examples == 0:
                return 0.0
            return total_score / successful_examples
        
        return evaluate_module
