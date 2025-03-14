import dspy
from typing import List, Callable, Any, Optional
from .agent import Agent
from .main import EvolutionaryOptimizer

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
        
        # Define evaluation function for agents
        def evaluate_module(agent: Agent) -> float:
            # Apply the agent's task chromosome as the module's prompt
            optimized_module.prompt = agent.chromosomes["task"]
            
            # Evaluate on training set
            total_score = 0.0
            num_examples = min(len(trainset), 10)  # Limit to 10 examples for efficiency
            
            for example in trainset[:num_examples]:
                try:
                    result = optimized_module(example)
                    score = metric(result, example)
                    total_score += score
                except Exception as e:
                    print(f"Error evaluating module: {e}")
                    return 0.0
            
            # Return average score
            return total_score / num_examples if num_examples > 0 else 0.0
        
        # Override the evaluation function
        optimizer.evaluate_agent = evaluate_module
        
        # Initialize population with a few random agents
        for i in range(10):
            # Create initial prompts with some variation
            initial_prompt = f"Instruction: Process the input and generate a response. Input: {{input}}"
            if i > 0:
                initial_prompt += f" Consider aspect {i} of the problem."
                
            agent = Agent(task_chromosome=initial_prompt)
            agent.score = optimizer.evaluate_agent(agent)
            optimizer.population.add_agent(agent)
            optimizer.statistics.update(agent)
        
        # Run optimization for a limited number of evaluations
        optimizer.running = True
        evaluation_count = 0
        
        while optimizer.running and evaluation_count < max_evaluations:
            # Select parents
            num_pairs = max(1, optimizer.num_parallel)
            parents = optimizer.population.get_candidates(num_pairs * 2)
            
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
