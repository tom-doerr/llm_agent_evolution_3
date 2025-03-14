import random
import threading
from typing import List, TypeVar, Dict, Any, Tuple

import toml

T = TypeVar('T')

# Thread-local storage for thread safety
thread_local = threading.local()

def _select_weighted_item(items: List[T], weights: List[float]) -> tuple[T, int]:
    """Select a single item based on weights and return the item and its index."""
    total_weight = sum(weights)
    if total_weight <= 0:
        # If all weights are zero, select randomly
        idx = random.randrange(len(items))
        return items[idx], idx
    
    # Weighted selection
    random_val = random.uniform(0, total_weight)
    cumulative_weight = 0
    for i, weight in enumerate(weights):
        cumulative_weight += weight
        if cumulative_weight >= random_val:
            return items[i], i
    
    # Fallback (should not reach here)
    return items[-1], len(items) - 1

def weighted_sample(items: List[T], weights: List[float], k: int = 1) -> List[T]:
    """Perform weighted sampling without replacement."""
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

        # Select an item and get its index
        selected_item, idx = _select_weighted_item(remaining_items, remaining_weights)
        
        # Add selected item to result
        result.append(selected_item)

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

def create_parent_pairs(parents: List[T]) -> List[Tuple[T, T]]:
    """Create pairs of parents for mating."""
    parent_pairs = []
    for i in range(0, len(parents), 2):
        if i+1 < len(parents):
            parent_pairs.append((parents[i], parents[i+1]))
    return parent_pairs

def save_to_toml(data: Dict[str, Any], filename: str) -> None:
    # Save data to TOML file
    with open(filename, 'w', encoding='utf-8') as file_handle:
        toml.dump(data, file_handle)

def load_from_toml(filename: str) -> Dict[str, Any]:
    # Load data from TOML file
    with open(filename, 'r', encoding='utf-8') as file_handle:
        return toml.load(file_handle)

def get_thread_rng():
    # Get a thread-local random number generator for thread safety
    if not hasattr(thread_local, 'rng'):
        thread_local.rng = random.Random()
    return thread_local.rng
