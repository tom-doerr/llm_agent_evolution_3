import statistics as stats
import time
import threading
from collections import deque
from typing import List, Dict, Any, Optional

# Try to import Rich components
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def update(self, agent: Agent) -> None:
        # Add new evaluation result (thread-safe)
        with self._lock:
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
        # Record which agents were selected for merging (thread-safe)
        with self._lock:
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
        # Calculate mean score (thread-safe)
        with self._lock:
            if not self.scores:
                return 0.0
            return stats.mean(self.scores)
    
    def get_median(self) -> float:
        # Calculate median score (thread-safe)
        with self._lock:
            if not self.scores:
                return 0.0
            return stats.median(self.scores)
    
    def get_std_dev(self) -> float:
        # Calculate standard deviation (thread-safe)
        with self._lock:
            if len(self.scores) < 2:
                return 0.0
            try:
                return stats.stdev(self.scores)
            except stats.StatisticsError:
                return 0.0
    
    def get_stats_dict(self, population_size: int = 0) -> Dict[str, Any]:
        # Get statistics as dictionary (thread-safe)
        with self._lock:
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
        
        if RICH_AVAILABLE:
            self._print_stats_rich(stats_dict, verbose)
        else:
            self._print_stats_plain(stats_dict, verbose)
    
    def _print_stats_rich(self, stats_dict: Dict[str, Any], verbose: bool) -> None:
        # Rich-formatted output
        console = Console()
        
        # Create a table for basic stats
        table = Table(title="Population Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Population size", str(stats_dict['population_size']))
        table.add_row("Total evaluations", str(stats_dict['total_evaluations']))
        table.add_row("Elapsed time", f"{stats_dict['elapsed_time']:.2f} seconds")
        table.add_row("Mean score", f"{stats_dict['mean']:.4f}")
        table.add_row("Median score", f"{stats_dict['median']:.4f}")
        table.add_row("Std deviation", f"{stats_dict['std_dev']:.4f}")
        
        if self.best_agent:
            table.add_row("Best agent", str(self.best_agent))
        
        console.print(table)
        
        # Print mating history if verbose
        if verbose and self.mating_history:
            self._print_mating_history_rich(console)
    
    def _print_mating_history_rich(self, console) -> None:
        # Print mating history with Rich
        mating_table = Table(title="Recent Mating Events")
        mating_table.add_column("#", style="dim")
        mating_table.add_column("Parent 1", style="cyan")
        mating_table.add_column("Parent 2", style="cyan")
        mating_table.add_column("Offspring", style="green")
        
        for i, event in enumerate(list(self.mating_history)[-5:]):
            mating_table.add_row(
                str(i+1),
                f"{event['parent1_score']:.4f}",
                f"{event['parent2_score']:.4f}",
                f"{event['offspring_score']:.4f}"
            )
        
        console.print(mating_table)
        
        # Show chromosomes of the most recent mating event if available
        if self.mating_history:
            self._print_chromosome_details_rich(console)
    
    def _print_chromosome_details_rich(self, console) -> None:
        # Print chromosome details with Rich
        latest = list(self.mating_history)[-1]
        parent1_id = latest['parent1']
        parent2_id = latest['parent2']
        
        # Find the agents by ID
        parent1 = self.best_agent if self.best_agent and self.best_agent.id == parent1_id else None
        parent2 = self.best_agent if self.best_agent and self.best_agent.id == parent2_id else None
        
        if parent1 or parent2:
            chromo_table = Table(title="Chromosome Details (Most Recent Mating)")
            chromo_table.add_column("Agent", style="cyan")
            chromo_table.add_column("Task Excerpt", style="green")
            chromo_table.add_column("Merging Excerpt", style="yellow")
            
            if parent1:
                chromo_table.add_row(
                    f"Parent 1 ({parent1.score:.4f})",
                    parent1.chromosomes['task'][:50] + "...",
                    parent1.chromosomes['merging'][:50] + "..."
                )
            if parent2:
                chromo_table.add_row(
                    f"Parent 2 ({parent2.score:.4f})",
                    parent2.chromosomes['task'][:50] + "...",
                    parent2.chromosomes['merging'][:50] + "..."
                )
            
            console.print(chromo_table)
    
    def _print_stats_plain(self, stats_dict: Dict[str, Any], verbose: bool) -> None:
        # Plain text output
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
            self._print_mating_history_plain()
    
    def _print_mating_history_plain(self) -> None:
        # Print mating history in plain text
        print("\n--- Recent Mating Events ---")
        for i, event in enumerate(list(self.mating_history)[-5:]):
            print(f"{i+1}. Parents: {event['parent1_score']:.4f} + {event['parent2_score']:.4f} → Offspring: {event['offspring_score']:.4f}")
            
        # Show chromosomes of the most recent mating event if available
        if self.mating_history:
            self._print_chromosome_details_plain()
    
    def _print_chromosome_details_plain(self) -> None:
        # Print chromosome details in plain text
        latest = list(self.mating_history)[-1]
        parent1_id = latest['parent1']
        parent2_id = latest['parent2']
        
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
        # Print basic stats first
        self.print_stats(verbose=True, population_size=population_size)
        
        if RICH_AVAILABLE:
            self._print_detailed_stats_rich()
        else:
            self._print_detailed_stats_plain()
    
    def _print_detailed_stats_rich(self) -> None:
        # Print detailed stats with Rich
        console = Console()
        
        # Print detailed best agent info
        if self.best_agent:
            best_table = Table(title="Best Agent Details")
            best_table.add_column("Property", style="cyan")
            best_table.add_column("Value", style="green")
            
            best_table.add_row("ID", self.best_agent.id)
            best_table.add_row("Score", f"{self.best_agent.score:.4f}")
            best_table.add_row("Task chromosome length", str(len(self.best_agent.chromosomes['task'])))
            
            console.print(best_table)
            
            # Print chromosomes in panels
            console.print(Panel(
                self.best_agent.chromosomes['task'][:500] + 
                ("..." if len(self.best_agent.chromosomes['task']) > 500 else ""),
                title="Task Chromosome",
                border_style="green"
            ))
            
            console.print(Panel(
                self.best_agent.chromosomes['merging'][:500] + 
                ("..." if len(self.best_agent.chromosomes['merging']) > 500 else ""),
                title="Merging Chromosome",
                border_style="yellow"
            ))
    
    def _print_detailed_stats_plain(self) -> None:
        # Print detailed stats in plain text
        print("\n--- Best Agent Details ---")
        if self.best_agent:
            print(f"ID: {self.best_agent.id}")
            print(f"Score: {self.best_agent.score:.4f}")
            print(f"Task chromosome length: {len(self.best_agent.chromosomes['task'])}")
            print(f"Task chromosome: {self.best_agent.chromosomes['task'][:100]}...")
            print(f"Merging chromosome: {self.best_agent.chromosomes['merging'][:100]}...")
