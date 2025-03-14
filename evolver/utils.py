import random
import toml
import threading
from typing import List, TypeVar, Dict, Any

T = TypeVar('T')

# Thread-local storage for thread safety
thread_local = threading.local()

def weighted_sample(items: List[T], weights: List[float], k: int = 1) -> List[T]:
    # Perform weighted sampling without replacement
    if not items or not weights or k <= 0:
        return []
    
    # Ensure k doesn't exceed the number of items
    k = min(k, len(items))
    
    # Make copies to avoid modifying the originals
    remaining_items = items.copy()
    remaining_weights = weights.copy()
    
    result = []
    for _ in range(k):
        if not remaining_items:
            break
            
        # Calculate total weight
        total_weight = sum(remaining_weights)
        if total_weight <= 0:
            # If all weights are zero, select randomly
            idx = random.randrange(len(remaining_items))
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
        
        # Add selected item to result
        result.append(remaining_items[idx])
        
        # Remove selected item from remaining options
        remaining_items.pop(idx)
        remaining_weights.pop(idx)
    
    return result

def prepare_weights(scores: List[float]) -> List[float]:
    # Prepare weights for selection (used by both parent selection and candidate selection)
    # Ensure all scores are positive
    min_score = min(scores) if scores else 0
    if min_score < 0:
        # Adjust scores to make them positive
        adjusted_scores = [score - min_score + 1 for score in scores]
        
        # Normalize so the highest adjusted score is 1.0
        max_adjusted = max(adjusted_scores)
        adjusted_scores = [score / max_adjusted for score in adjusted_scores]
    else:
        adjusted_scores = [max(score, 0.0001) for score in scores]
    
    # Calculate weights as score^2 (Pareto distribution)
    return [score * score for score in adjusted_scores]

def save_to_toml(data: Dict[str, Any], filename: str) -> None:
    # Save data to TOML file
    with open(filename, 'w') as f:
        toml.dump(data, f)

def load_from_toml(filename: str) -> Dict[str, Any]:
    # Load data from TOML file
    with open(filename, 'r') as f:
        return toml.load(f)

def get_thread_rng():
    # Get a thread-local random number generator for thread safety
    if not hasattr(thread_local, 'rng'):
        thread_local.rng = random.Random()
    return thread_local.rng
