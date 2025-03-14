import threading
import time
from collections import deque
from typing import Dict, Any, Optional
import statistics as stats

from .agent import Agent
from .constants import STATS_WINDOW_SIZE

# Try to import Rich components
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

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
        
        # TODO: Split this class into smaller components
        # TODO: Add more metrics for evolution quality
        # TODO: Implement better visualization of population diversity
    
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
        """Print current statistics."""
        # Get statistics
        stats_dict = self.get_stats_dict(population_size)
        
        # Format and print statistics using the appropriate formatter
        self._format_and_print_stats(stats_dict, verbose)
    
    def _format_and_print_stats(self, stats_dict: Dict[str, Any], verbose: bool) -> None:
        """Format and print statistics using the appropriate formatter."""
        # Use rich if available, otherwise plain text
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
        
        # Add rows for each statistic
        for metric, value in [
            ("Population size", str(stats_dict['population_size'])),
            ("Total evaluations", str(stats_dict['total_evaluations'])),
            ("Elapsed time", f"{stats_dict['elapsed_time']:.2f} seconds"),
            ("Mean score", f"{stats_dict['mean']:.4f}"),
            ("Median score", f"{stats_dict['median']:.4f}"),
            ("Std deviation", f"{stats_dict['std_dev']:.4f}"),
        ]:
            table.add_row(metric, value)
        
        if self.best_agent:
            table.add_row("Best agent", str(self.best_agent))
        
        console.print(table)
        
        # Print mating history if verbose
        if verbose and self.mating_history:
            self._print_mating_history(console)
    
    def _print_mating_history(self, console=None) -> None:
        # Print mating history (works with both rich and plain text)
        if not self.mating_history:
            return
            
        if console and RICH_AVAILABLE:
            # Rich version
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
            
            # Show chromosomes of the most recent mating event
            self._print_chromosome_details(console)
        else:
            # Plain text version
            print("\n--- Recent Mating Events ---")
            for i, event in enumerate(list(self.mating_history)[-5:]):
                print(f"{i+1}. Parents: {event['parent1_score']:.4f} + {event['parent2_score']:.4f} → Offspring: {event['offspring_score']:.4f}")
            
            # Show chromosomes of the most recent mating event
            self._print_chromosome_details()
    
    def _print_chromosome_details(self, console=None) -> None:
        # Print chromosome details (works with both rich and plain text)
        if not self.mating_history:
            return
            
        latest = list(self.mating_history)[-1]
        parent1_id = latest['parent1']
        parent2_id = latest['parent2']
        
        # Find the agents by ID
        parent1 = self.best_agent if self.best_agent and self.best_agent.id == parent1_id else None
        parent2 = self.best_agent if self.best_agent and self.best_agent.id == parent2_id else None
        
        if not (parent1 or parent2):
            return
            
        if console and RICH_AVAILABLE:
            # Rich version
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
        else:
            # Plain text version
            print("\n--- Chromosome Details (Most Recent Mating) ---")
            if parent1:
                print(f"Parent 1 ({parent1.score:.4f}) Task: {parent1.chromosomes['task'][:50]}...")
                print(f"Parent 1 Merging: {parent1.chromosomes['merging'][:50]}...")
            if parent2:
                print(f"Parent 2 ({parent2.score:.4f}) Task: {parent2.chromosomes['task'][:50]}...")
                print(f"Parent 2 Merging: {parent2.chromosomes['merging'][:50]}...")
    
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
            self._print_mating_history()
    
    def print_detailed_stats(self, population_size: int = 0) -> None:
        # Print detailed statistics (for exit)
        # Print basic stats first
        self.print_stats(verbose=True, population_size=population_size)
        
        # Print details for best, worst, and median agents
        self._print_agent_details(self.best_agent, "Best")
        self._print_agent_details(self.worst_agent, "Worst")
        self._print_agent_details(self._get_median_agent(), "Median")
        
        # Print mating history
        if self.mating_history:
            self._print_mating_history()
    
    def _get_median_agent(self) -> Optional[Agent]:
        """Get the agent with the median score."""
        with self._lock:
            if not self.scores:
                return None
            
            # Find the median score
            median_score = self.get_median()
            
            # Find the agent closest to the median score
            closest_agent = None
            closest_distance = float('inf')
            
            # We don't have direct access to all agents, so use best and worst as fallback
            agents_to_check = []
            if self.best_agent:
                agents_to_check.append(self.best_agent)
            if self.worst_agent and self.worst_agent != self.best_agent:
                agents_to_check.append(self.worst_agent)
                
            for agent in agents_to_check:
                distance = abs(agent.score - median_score)
                if distance < closest_distance:
                    closest_agent = agent
                    closest_distance = distance
            
            return closest_agent
    
    def _print_agent_details(self, agent: Optional[Agent], label: str) -> None:
        # Print details for a specific agent
        if not agent:
            return
            
        if RICH_AVAILABLE:
            console = Console()
            table = Table(title=f"{label} Agent Details")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            # Add rows for each property
            for prop, value in [
                ("ID", agent.id),
                ("Score", f"{agent.score:.4f}"),
                ("Task chromosome length", str(len(agent.chromosomes['task']))),
                ("Merging chromosome length", str(len(agent.chromosomes['merging'])))
            ]:
                table.add_row(prop, value)
            
            console.print(table)
            
            # Show chromosome content
            console.print(Panel(
                agent.chromosomes['task'][:500] + 
                ("..." if len(agent.chromosomes['task']) > 500 else ""),
                title=f"{label} Task Chromosome",
                border_style="green"
            ))
        else:
            print(f"\n--- {label} Agent Details ---")
            print(f"ID: {agent.id}")
            print(f"Score: {agent.score:.4f}")
            print(f"Task chromosome length: {len(agent.chromosomes['task'])}")
            print(f"Merging chromosome length: {len(agent.chromosomes['merging'])}")
            print(f"Task chromosome: {agent.chromosomes['task'][:100]}...")
            print(f"Merging chromosome: {agent.chromosomes['merging'][:100]}...")
    
    
    def log_optimization_progress(self, population_size: int, verbose: bool = False) -> None:
        # Log detailed optimization progress
        stats = self.get_stats_dict(population_size)
        
        print(f"\n--- Optimization Progress (Evaluation {self.total_evaluations}) ---")
        print(f"Population: {population_size} agents | Evaluations: {self.total_evaluations}")
        print(f"Scores: Mean={stats['mean']:.2f} | Median={stats['median']:.2f} | Best={stats['best']:.2f}")
        
        if self.best_agent:
            # Show the actual output (which is the chromosome in this simple case)
            output = self.best_agent.chromosomes['task']
            count_a = output.count('a')
            penalty = max(0, len(output) - 23)  # Using TEST_OPTIMAL_LENGTH
            
            print(f"\nBest agent output excerpt: \"{output[:30]}{'...' if len(output) > 30 else ''}\"")
            print(f"Score: {self.best_agent.score:.2f} ({count_a} 'a's, {penalty} char penalty)")
        
        if verbose and self.mating_history and len(self.mating_history) > 0:
            latest = list(self.mating_history)[-1]
            print(f"\nRecent mating: {latest['parent1_score']:.2f} + {latest['parent2_score']:.2f} → {latest['offspring_score']:.2f}")
