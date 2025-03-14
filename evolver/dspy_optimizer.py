import dspy
from typing import Any, Dict, List, Optional, Callable, Union
from .main import EvolutionaryOptimizer
from .agent import Agent

class DSPyOptimizer:
    """
    DSPy optimizer interface for the evolutionary algorithm.
    
    This class provides an interface to use the evolutionary algorithm
    as an optimizer for DSPy pipelines.
    """
    
    def __init__(self, 
                 max_agents: int = 1_000_000,
                 parallel: int = 10,
                 verbose: bool = False,
                 model_name: str = "openrouter/google/gemini-2.0-flash-001"):
        """
        Initialize the DSPy optimizer.
        
        Args:
            max_agents: Maximum population size
            parallel: Number of agents to run in parallel
            verbose: Enable verbose output
            model_name: LLM model to use
        """
        self.args = {
            "max_agents": max_agents,
            "parallel": parallel,
            "verbose": verbose,
            "model": model_name,
        }
        self.optimizer = None
        
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
        # Initialize optimizer if not already initialized
        if not self.optimizer:
            self.optimizer = EvolutionaryOptimizer(self.args)
        
        # Create evaluation function
        def evaluate_module(agent: Agent) -> float:
            # Extract prompt from agent's task chromosome
            prompt = agent.chromosomes["task"]
            
            # Apply prompt to module
            try:
                # Create a copy of the module to avoid modifying the original
                module_copy = module.copy()
                
                # Apply prompt to module (implementation depends on module type)
                if hasattr(module_copy, "set_prompt"):
                    module_copy.set_prompt(prompt)
                elif hasattr(module_copy, "prompt"):
                    module_copy.prompt = prompt
                
                # Evaluate on training set
                scores = []
                for example in trainset[:min(len(trainset), 10)]:  # Use subset for efficiency
                    try:
                        result = module_copy(example)
                        score = metric(result, example)
                        scores.append(score)
                    except Exception as e:
                        scores.append(0.0)  # Penalty for errors
                
                # Calculate average score
                avg_score = sum(scores) / len(scores) if scores else 0.0
                return avg_score
                
            except Exception as e:
                return 0.0  # Penalty for errors
        
        # Override evaluation function
        self.optimizer.evaluate_agent = evaluate_module
        
        # Run optimization for specified number of evaluations
        self.optimizer.running = True
        while self.optimizer.running and self.optimizer.statistics.total_evaluations < max_evaluations:
            # Select parents
            num_pairs = max(1, self.optimizer.num_parallel)
            parents = select_parents(self.optimizer.population.agents, num_pairs * 2)
            
            # Create parent pairs
            parent_pairs = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    parent_pairs.append((parents[i], parents[i+1]))
            
            # Create and evaluate offspring
            for pair in parent_pairs:
                if not self.optimizer.running:
                    break
                    
                offspring = self.optimizer.create_and_evaluate_offspring(pair)
                if offspring:
                    self.optimizer.population.add_agent(offspring)
            
            # Print statistics periodically
            if self.optimizer.statistics.total_evaluations % 10 == 0:
                self.optimizer.statistics.print_stats(
                    verbose=self.optimizer.verbose,
                    population_size=len(self.optimizer.population)
                )
        
        # Get best agent
        best_agent = self.optimizer.statistics.best_agent
        
        # Apply best prompt to original module
        if best_agent:
            best_prompt = best_agent.chromosomes["task"]
            if hasattr(module, "set_prompt"):
                module.set_prompt(best_prompt)
            elif hasattr(module, "prompt"):
                module.prompt = best_prompt
        
        return module
