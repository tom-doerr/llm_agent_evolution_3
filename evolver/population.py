import random
import toml
from typing import List, Dict, Any, Optional
from .agent import Agent
from .constants import MAX_POPULATION_SIZE

class Population:
    def __init__(self, max_size: int = MAX_POPULATION_SIZE):
        self.agents = []
        self.max_size = max_size
    
    def add_agent(self, agent: Agent) -> None:
        # Add new agent to population
        self.agents.append(agent)
        if len(self.agents) > self.max_size:
            self.prune()
    
    def remove_agent(self, agent: Agent) -> None:
        # Remove agent from population
        if agent in self.agents:
            self.agents.remove(agent)
    
    def get_best_agents(self, n: int = 1) -> List[Agent]:
        # Get top N agents by score
        sorted_agents = sorted(self.agents, key=lambda a: a.score, reverse=True)
        return sorted_agents[:min(n, len(sorted_agents))]
    
    def get_candidates(self, n: int, weights: Optional[List[float]] = None) -> List[Agent]:
        # Weighted sampling without replacement
        if not weights:
            weights = [agent.score for agent in self.agents]
            # Ensure all weights are positive
            min_weight = min(weights) if weights else 0
            if min_weight < 0:
                weights = [w - min_weight + 1 for w in weights]
        
        # Ensure n doesn't exceed population size
        n = min(n, len(self.agents))
        
        if n == 0 or not self.agents:
            return []
        
        # Perform weighted sampling without replacement
        candidates = []
        remaining_agents = self.agents.copy()
        remaining_weights = weights.copy()
        
        for _ in range(n):
            if not remaining_agents:
                break
                
            # Select an agent based on weights
            total_weight = sum(remaining_weights)
            if total_weight <= 0:
                # If all weights are zero, select randomly
                idx = random.randrange(len(remaining_agents))
            else:
                # Weighted selection
                r = random.uniform(0, total_weight)
                cumulative_weight = 0
                idx = 0
                for i, weight in enumerate(remaining_weights):
                    cumulative_weight += weight
                    if cumulative_weight >= r:
                        idx = i
                        break
            
            # Add selected agent to candidates
            candidates.append(remaining_agents[idx])
            
            # Remove selected agent from remaining options
            remaining_agents.pop(idx)
            remaining_weights.pop(idx)
        
        return candidates
    
    def prune(self) -> None:
        # Remove worst agents when population exceeds limit
        if len(self.agents) <= self.max_size:
            return
            
        # Sort by score (descending)
        self.agents.sort(key=lambda a: a.score, reverse=True)
        
        # Keep only the best agents
        self.agents = self.agents[:self.max_size]
    
    def save(self, filename: str) -> None:
        # Save population to TOML file
        data = {
            "max_size": self.max_size,
            "agents": [agent.to_dict() for agent in self.agents]
        }
        with open(filename, 'w') as f:
            toml.dump(data, f)
    
    def load(self, filename: str) -> None:
        # Load population from TOML file
        with open(filename, 'r') as f:
            data = toml.load(f)
        
        self.max_size = data.get("max_size", MAX_POPULATION_SIZE)
        self.agents = [Agent.from_dict(agent_data) for agent_data in data.get("agents", [])]
    
    def __len__(self) -> int:
        return len(self.agents)
