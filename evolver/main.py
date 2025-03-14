import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple, Optional

from .agent import Agent
from .cli import parse_args
from .constants import DEFAULT_PARALLEL_AGENTS
from .evolution import select_parents, create_offspring, external_command_evaluation
from .utils import create_parent_pairs
from .llm import LLMInterface
from .population import Population
from .statistics import Statistics

class EvolutionaryOptimizer:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.population = Population(max_size=args.get("max_agents", 1_000_000))
        self.statistics = Statistics()
        self.llm = LLMInterface(model_name=args.get("model", "openrouter/google/gemini-2.0-flash-001"))
        self.llm.initialize()
        self.verbose = args.get("verbose", False)
        self.running = True
        self.eval_command = args.get("eval_command", "")
        self.num_parallel = args.get("parallel", DEFAULT_PARALLEL_AGENTS)
        self._shutdown_lock = threading.Lock()
        
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.handle_interrupt)
    
    def handle_interrupt(self, *_):
        # Handle Ctrl+C for graceful shutdown (thread-safe)
        with self._shutdown_lock:
            if self.running:
                print("\n\nShutting down gracefully. Please wait...")
                self.running = False
                # Print detailed statistics on exit
                self.statistics.print_detailed_stats(population_size=len(self.population))
    
    def evaluate_agent(self, agent: Agent) -> float:
        # Evaluate agent using external command or test function
        if self.eval_command:
            return external_command_evaluation(agent, self.eval_command)
        
        # Default test task: reward for 'a's up to optimal length
        from .constants import TEST_OPTIMAL_LENGTH
        text = agent.chromosomes["task"]
        count_a = text.count('a')
        penalty = max(0, len(text) - TEST_OPTIMAL_LENGTH)
        return count_a - penalty
    
    def create_and_evaluate_offspring(self, parents: Tuple[Agent, Agent]) -> Optional[Agent]:
        # Create and evaluate a new agent from parents
        parent1, parent2 = parents
        
        # Create offspring using LLM for combination
        offspring = create_offspring(parent1, parent2, self.llm)
        
        # Evaluate offspring
        score = self.evaluate_agent(offspring)
        offspring.score = score
        
        # Track mating for statistics
        self.statistics.track_mating(parent1, parent2, offspring)
        
        # Update statistics
        self.statistics.update(offspring)
        
        if self.verbose:
            print(f"New agent: {offspring}, Parents: {parent1.score:.4f} + {parent2.score:.4f}")
            print(f"Task excerpt: {offspring.chromosomes['task'][:30]}...")
        
        return offspring
    
    
    def run_iteration(self):
        """Run a single iteration of the evolutionary algorithm."""
        # Select parents
        num_pairs = max(1, self.num_parallel)
        parents = self._select_parents_for_iteration(num_pairs * 2)
        
        # Create parent pairs
        parent_pairs = create_parent_pairs(parents)
        
        # Process pairs
        self._process_parent_pairs(parent_pairs)
    
    def _select_parents_for_iteration(self, num_parents):
        """Select parents for the current iteration."""
        return select_parents(self.population.agents, num_parents)
    
    def _process_parent_pairs(self, parent_pairs):
        """Process parent pairs in parallel to create offspring."""
        with ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            # Submit tasks
            future_to_pair = {
                executor.submit(self.create_and_evaluate_offspring, pair): pair
                for pair in parent_pairs
            }
            
            # Process results
            for future in as_completed(future_to_pair):
                if not self.running:
                    break
                    
                try:
                    new_agent = future.result()
                    if new_agent:
                        self.population.add_agent(new_agent)
                except (ValueError, TypeError, RuntimeError) as error:
                    print(f"Error processing offspring: {error}")
                except Exception as error:
                    print(f"Unexpected error processing offspring: {error}")
    
    def _initialize_population(self):
        # Initialize population with a few random agents if empty
        if len(self.population) == 0:
            for i in range(10):
                agent = Agent(task_chromosome="a" * ((i % 5) + 1))
                agent.score = self.evaluate_agent(agent)
                self.population.add_agent(agent)
                self.statistics.update(agent)
    
    def _process_stdin_input(self):
        # Check for input from stdin if available
        if not sys.stdin.isatty():
            stdin_input = sys.stdin.read().strip()
            if stdin_input:
                # Create an agent from stdin input
                agent = Agent(task_chromosome=stdin_input)
                agent.score = self.evaluate_agent(agent)
                self.population.add_agent(agent)
                self.statistics.update(agent)
                print(f"Created agent from stdin with score: {agent.score:.4f}")
    
    def _load_population(self):
        # Load agent if specified
        if self.args.get("load"):
            try:
                self.population.load(self.args["load"])
                print(f"Loaded population from {self.args['load']}")
            except (FileNotFoundError, PermissionError) as error:
                print(f"File error loading population: {error}")
            except Exception as error:
                print(f"Unexpected error loading population: {error}")
    
    def run(self):
        # Process input and initialize population
        self._process_stdin_input()
        self._initialize_population()
        self._load_population()
        
        print(f"Starting evolution with {self.num_parallel} parallel agents")
        
        # Main evolution loop
        stats_interval = 10  # Print stats every N iterations
        iteration = 0
        
        while self.running:
            iteration += 1
            
            # Run one iteration
            self.run_iteration()
            
            # Print statistics periodically
            if iteration % stats_interval == 0:
                self.statistics.print_stats(verbose=self.verbose, population_size=len(self.population))
            
            # Small delay to prevent CPU hogging
            time.sleep(0.01)
        
        # Print final statistics
        print("\n=== Final Statistics ===")
        self.statistics.print_detailed_stats(population_size=len(self.population))
        
        # Save best agent if specified
        if self.args.get("save"):
            try:
                best_agent = self.statistics.best_agent
                if best_agent:
                    filename = self.args["save"]
                    self.population.save(filename)
                    print(f"Saved population to {filename}")
            except Exception as error:
                print(f"Error saving population: {error}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create and run optimizer
    optimizer = EvolutionaryOptimizer(args)
    optimizer.run()

if __name__ == "__main__":
    main()
