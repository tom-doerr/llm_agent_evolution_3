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
            parents = self.optimizer.population.get_candidates(num_pairs * 2)
            
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
import dspy
from typing import List, Callable, Any, Optional
from .agent import Agent
from .main import EvolutionaryOptimizer

class DSPyOptimizer:
    def __init__(self, 
                 max_agents: int = 1_000_000,
                 parallel: int = 10,
                 verbose: bool = False):
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
            for example in trainset[:min(len(trainset), 10)]:  # Limit to 10 examples for efficiency
                try:
                    result = optimized_module(example)
                    score = metric(result, example)
                    total_score += score
                except Exception as e:
                    print(f"Error evaluating module: {e}")
                    return 0.0
            
            # Return average score
            return total_score / min(len(trainset), 10)
        
        # Override the evaluation function
        optimizer.evaluate_agent = evaluate_module
        
        # Initialize population with a few random agents
        for _ in range(10):
            agent = Agent(task_chromosome=f"Instruction: Process the input and generate a response. Input: {{input}}")
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
                optimizer.statistics.print_stats(verbose=optimizer.verbose, population_size=len(optimizer.population))
        
        # Get the best agent and apply its chromosome to the module
        best_agent = optimizer.statistics.best_agent
        if best_agent:
            optimized_module.prompt = best_agent.chromosomes["task"]
        
        return optimized_module
import dspy
from typing import List, Callable, Any, Optional
from .agent import Agent
from .main import EvolutionaryOptimizer

class DSPyOptimizer:
    def __init__(self, 
                 max_agents: int = 1_000_000,
                 parallel: int = 10,
                 verbose: bool = False):
        # Initialize the DSPy optimizer
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
            for example in trainset[:min(len(trainset), 10)]:  # Limit to 10 examples for efficiency
                try:
                    result = optimized_module(example)
                    score = metric(result, example)
                    total_score += score
                except Exception as e:
                    print(f"Error evaluating module: {e}")
                    return 0.0
            
            # Return average score
            return total_score / min(len(trainset), 10)
        
        # Override the evaluation function
        optimizer.evaluate_agent = evaluate_module
        
        # Initialize population with a few random agents
        for _ in range(10):
            agent = Agent(task_chromosome=f"Instruction: Process the input and generate a response. Input: {{input}}")
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
                optimizer.statistics.print_stats(verbose=optimizer.verbose, population_size=len(optimizer.population))
        
        # Get the best agent and apply its chromosome to the module
        best_agent = optimizer.statistics.best_agent
        if best_agent:
            optimized_module.prompt = best_agent.chromosomes["task"]
        
        return optimized_module
import dspy
from typing import List, Callable, Any, Optional, Dict
from .agent import Agent
from .main import EvolutionaryOptimizer
from .evolution import select_parents

class DSPyOptimizer:
    """
    DSPy optimizer interface for the evolutionary algorithm.
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
            for example in trainset[:min(len(trainset), 10)]:  # Limit to 10 examples for efficiency
                try:
                    result = optimized_module(example)
                    score = metric(result, example)
                    total_score += score
                except Exception as e:
                    print(f"Error evaluating module: {e}")
                    return 0.0
            
            # Return average score
            return total_score / min(len(trainset), 10)
        
        # Override the evaluation function
        optimizer.evaluate_agent = evaluate_module
        
        # Initialize population with a few random agents
        for _ in range(10):
            agent = Agent(task_chromosome=f"Instruction: Process the input and generate a response. Input: {{input}}")
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
                optimizer.statistics.print_stats(verbose=optimizer.verbose, population_size=len(optimizer.population))
        
        # Get the best agent and apply its chromosome to the module
        best_agent = optimizer.statistics.best_agent
        if best_agent:
            optimized_module.prompt = best_agent.chromosomes["task"]
        
        return optimized_module
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
