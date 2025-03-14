import random
import toml
from typing import List, TypeVar, Dict, Any

T = TypeVar('T')

def weighted_sample(items: List[T], weights: List[float], k: int = 1) -> List[T]:
    """
    Perform weighted sampling without replacement.
    
    Args:
        items: List of items to sample from
        weights: List of weights corresponding to items
        k: Number of items to sample
        
    Returns:
        List of sampled items
    """
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

def save_to_toml(data: Dict[str, Any], filename: str) -> None:
    """
    Save data to TOML file.
    
    Args:
        data: Dictionary to save
        filename: Path to save file
    """
    with open(filename, 'w') as f:
        toml.dump(data, f)

def load_from_toml(filename: str) -> Dict[str, Any]:
    """
    Load data from TOML file.
    
    Args:
        filename: Path to load file
        
    Returns:
        Dictionary with loaded data
    """
    with open(filename, 'r') as f:
        return toml.load(f)
