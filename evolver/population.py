import threading
from typing import List, Optional
import toml
from .agent import Agent
from .constants import MAX_POPULATION_SIZE
from .utils import weighted_sample, prepare_weights

class Population:
    def __init__(self, max_size: int = MAX_POPULATION_SIZE):
        self.agents = []
        self.max_size = max_size
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def add_agent(self, agent: Agent) -> None:
        # Add new agent to population (thread-safe)
        with self._lock:
            self.agents.append(agent)
            if len(self.agents) > self.max_size:
                self.prune()
    
    def remove_agent(self, agent: Agent) -> None:
        # Remove agent from population (thread-safe)
        with self._lock:
            if agent in self.agents:
                self.agents.remove(agent)
    
    def get_best_agents(self, n: int = 1) -> List[Agent]:
        # Get top N agents by score (thread-safe)
        with self._lock:
            sorted_agents = sorted(self.agents, key=lambda a: a.score, reverse=True)
            return sorted_agents[:min(n, len(sorted_agents))]
    
    def get_candidates(self, n: int, weights: Optional[List[float]] = None) -> List[Agent]:
        # Weighted sampling without replacement (thread-safe)
        with self._lock:
            if not weights:
                weights = prepare_weights([agent.score for agent in self.agents])
            
            return weighted_sample(self.agents, weights, n)
    
    def prune(self) -> None:
        # Remove worst agents when population exceeds limit (thread-safe)
        with self._lock:
            if len(self.agents) <= self.max_size:
                return
                
            # Sort by score (descending)
            self.agents.sort(key=lambda a: a.score, reverse=True)
            
            # Keep only the best agents
            self.agents = self.agents[:self.max_size]
    
    def save(self, filename: str) -> None:
        # Save population to TOML file (thread-safe)
        with self._lock:
            data = {
                "max_size": self.max_size,
                "agents": [agent.to_dict() for agent in self.agents]
            }
            with open(filename, 'w', encoding='utf-8') as file_handle:
                toml.dump(data, file_handle)
    
    def load(self, filename: str) -> None:
        # Load population from TOML file (thread-safe)
        with self._lock:
            with open(filename, 'r', encoding='utf-8') as file_handle:
                data = toml.load(file_handle)
            
            self.max_size = data.get("max_size", MAX_POPULATION_SIZE)
            self.agents = [Agent.from_dict(agent_data) for agent_data in data.get("agents", [])]
    
    def __len__(self) -> int:
        return len(self.agents)
