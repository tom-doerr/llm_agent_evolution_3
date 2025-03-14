import uuid
from typing import Dict, Any, Optional

class Agent:
    def __init__(self, 
                 task_chromosome: str = "", 
                 merging_chromosome: str = "",
                 score: float = 0.0,
                 agent_id: Optional[str] = None):
        # Initialize chromosomes
        self.chromosomes = {
            "task": task_chromosome,
            "merging": merging_chromosome
        }
        self.score = score
        self.id = agent_id or str(uuid.uuid4())
        self.creation_timestamp = uuid.uuid1().time
    
    def to_dict(self) -> Dict[str, Any]:
        # Convert agent to dictionary for serialization
        return {
            "id": self.id,
            "chromosomes": self.chromosomes,
            "score": self.score,
            "creation_timestamp": self.creation_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        # Create agent from dictionary
        agent = cls(
            task_chromosome=data["chromosomes"]["task"],
            merging_chromosome=data["chromosomes"]["merging"],
            score=data["score"],
            agent_id=data["id"]
        )
        agent.creation_timestamp = data.get("creation_timestamp", agent.creation_timestamp)
        return agent
    
    def __str__(self) -> str:
        # String representation for debugging
        return f"Agent(id={self.id}, score={self.score:.4f}, task_len={len(self.chromosomes['task'])})"
