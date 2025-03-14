import statistics as stats
from typing import List, Dict, Any, Optional
from collections import deque
import time

from .agent import Agent
from .constants import STATS_WINDOW_SIZE

class Statistics:
    def __init__(self, window_size: int = STATS_WINDOW_SIZE):
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)
        self.best_agent: Optional[Agent] = None
        self.worst_agent: Optional[Agent] = None
        self.mating_history = deque(maxlen=window_size)
        self.start_time = time.time()
        self.total_evaluations = 0
    
    def update(self, agent: Agent) -> None:
        # Add new evaluation result
        self.scores.append(agent.score)
        self.total_evaluations += 1
        
        # Update best agent
        if self.best_agent is None or agent.score > self.best_agent.score:
            self.best_agent = agent
        
        # Update worst agent in current window
        if len(self.scores) == 1:
            self.worst_agent = agent
        elif agent.score < self.worst_agent.score:
            self.worst_agent = agent
    
    def track_mating(self, parent1: Agent, parent2: Agent, offspring: Agent) -> None:
        # Record which agents were selected for merging
        self.mating_history.append({
            "parent1": parent1.id,
            "parent2": parent2.id,
            "parent1_score": parent1.score,
            "parent2_score": parent2.score,
            "offspring": offspring.id,
            "offspring_score": offspring.score,
            "timestamp": time.time()
        })
    
    def get_mean(self) -> float:
        # Calculate mean score
        if not self.scores:
            return 0.0
        return stats.mean(self.scores)
    
    def get_median(self) -> float:
        # Calculate median score
        if not self.scores:
            return 0.0
        return stats.median(self.scores)
    
    def get_std_dev(self) -> float:
        # Calculate standard deviation
        if len(self.scores) < 2:
            return 0.0
        try:
            return stats.stdev(self.scores)
        except stats.StatisticsError:
            return 0.0
    
    def get_stats_dict(self, population_size: int = 0) -> Dict[str, Any]:
        # Get statistics as dictionary
        return {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "std_dev": self.get_std_dev(),
            "best": max(self.scores) if self.scores else 0.0,
            "worst": min(self.scores) if self.scores else 0.0,
            "total_evaluations": self.total_evaluations,
            "elapsed_time": time.time() - self.start_time,
            "population_size": population_size
        }
    
    def print_stats(self, verbose: bool = False, population_size: int = 0) -> None:
        # Print current statistics
        stats_dict = self.get_stats_dict(population_size)
        
        print("\n--- Population Statistics ---")
        print(f"Population size: {stats_dict['population_size']}")
        print(f"Total evaluations: {stats_dict['total_evaluations']}")
        print(f"Elapsed time: {stats_dict['elapsed_time']:.2f} seconds")
        print(f"Mean score: {stats_dict['mean']:.4f}")
        print(f"Median score: {stats_dict['median']:.4f}")
        print(f"Std deviation: {stats_dict['std_dev']:.4f}")
        
        if self.best_agent:
            print(f"Best agent: {self.best_agent}")
        
        if verbose and self.mating_history:
            print("\n--- Recent Mating Events ---")
            for i, event in enumerate(list(self.mating_history)[-5:]):
                print(f"{i+1}. Parents: {event['parent1_score']:.4f} + {event['parent2_score']:.4f} → Offspring: {event['offspring_score']:.4f}")
                
            # Show chromosomes of the most recent mating event if available
            if self.mating_history:
                latest = list(self.mating_history)[-1]
                parent1_id = latest['parent1']
                parent2_id = latest['parent2']
                offspring_id = latest['offspring']
                
                # Find the agents by ID
                parent1 = self.best_agent if self.best_agent and self.best_agent.id == parent1_id else None
                parent2 = self.best_agent if self.best_agent and self.best_agent.id == parent2_id else None
                
                if parent1 or parent2:
                    print("\n--- Chromosome Details (Most Recent Mating) ---")
                    if parent1:
                        print(f"Parent 1 Task (excerpt): {parent1.chromosomes['task'][:50]}...")
                        print(f"Parent 1 Merging (excerpt): {parent1.chromosomes['merging'][:50]}...")
                    if parent2:
                        print(f"Parent 2 Task (excerpt): {parent2.chromosomes['task'][:50]}...")
                        print(f"Parent 2 Merging (excerpt): {parent2.chromosomes['merging'][:50]}...")
    
    def print_detailed_stats(self, population_size: int = 0) -> None:
        # Print detailed statistics (for exit)
        self.print_stats(verbose=True, population_size=population_size)
        
        print("\n--- Best Agent Details ---")
        if self.best_agent:
            print(f"ID: {self.best_agent.id}")
            print(f"Score: {self.best_agent.score:.4f}")
            print(f"Task chromosome length: {len(self.best_agent.chromosomes['task'])}")
            print(f"Task chromosome: {self.best_agent.chromosomes['task'][:100]}...")
            print(f"Merging chromosome: {self.best_agent.chromosomes['merging'][:100]}...")
