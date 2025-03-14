import string
from typing import List, Tuple, Optional, TypeVar

from .agent import Agent
from .constants import MAX_CHROMOSOME_LENGTH
from .utils import weighted_sample, prepare_weights, create_parent_pairs

T = TypeVar('T')


def select_parents(population: List[Agent], num_parents: int) -> List[Agent]:
    # Select parents using Pareto distribution
    if not population or num_parents <= 0:
        return []
    
    # Ensure we don't try to select more parents than available
    num_to_select = min(num_parents, len(population))
    
    scores = [agent.score for agent in population]
    weights = prepare_weights(scores)
    
    # Use common weighted sampling utility
    return weighted_sample(population, weights, num_to_select)

def find_hotspots(text: str) -> List[int]:
    # Identify punctuation and space positions as potential crossover points
    hotspots = []
    
    # Add punctuation positions
    for i, char in enumerate(text):
        if char in string.punctuation:
            hotspots.append(i)
    
    # Add some space positions with a certain probability
    from .utils import get_thread_rng
    rng = get_thread_rng()
    space_hotspot_probability = 0.3
    for i, char in enumerate(text):
        if char == ' ' and rng.random() < space_hotspot_probability:
            hotspots.append(i)
    
    return sorted(hotspots)

def combine_chromosomes(parent1: Agent, parent2: Agent, chromosome_name: str) -> str:
    # Combine chromosomes with hotspot-based crossover
    chromosome1 = parent1.chromosomes.get(chromosome_name, "")
    chromosome2 = parent2.chromosomes.get(chromosome_name, "")
    
    # If either chromosome is empty, return the other
    if not chromosome1:
        return chromosome2
    if not chromosome2:
        return chromosome1
    
    # Find potential crossover points (hotspots)
    hotspots1 = find_hotspots(chromosome1)
    hotspots2 = find_hotspots(chromosome2)
    
    # If no hotspots found, create some arbitrary ones
    if not hotspots1:
        # Create crossover points roughly every 5-10 characters
        from .utils import get_thread_rng
        rng = get_thread_rng()
        avg_segment_length = rng.randint(5, 10)
        for i in range(avg_segment_length, len(chromosome1), avg_segment_length):
            if i < len(chromosome1):
                hotspots1.append(i)
    
    # Calculate crossover probability to achieve on average one switch per chromosome
    crossover_probability = 1.0 / (len(hotspots1) + 1) if hotspots1 else 0.5
    
    # Perform crossover
    result = ""
    current_parent = 1  # Start with parent1
    last_pos = 0
    
    from .utils import get_thread_rng
    rng = get_thread_rng()
    
    # Determine if we should use a different strategy based on scores
    score_ratio = parent1.score / max(0.0001, parent2.score)
    
    # If one parent is significantly better, bias toward that parent
    if score_ratio > 2.0:  # Parent1 is much better
        bias = 0.7  # 70% chance to keep parent1's genes
    elif score_ratio < 0.5:  # Parent2 is much better
        bias = 0.3  # 30% chance to keep parent1's genes
    else:
        bias = 0.5  # Equal chance
    
    for pos in sorted(hotspots1):
        if pos > len(chromosome1):
            break
            
        if rng.random() < crossover_probability:
            # Switch parents at this hotspot, with bias
            if current_parent == 1:
                result += chromosome1[last_pos:pos]
                current_parent = 2 if rng.random() > bias else 1
            else:
                result += chromosome2[last_pos:min(pos, len(chromosome2))]
                current_parent = 1 if rng.random() > (1 - bias) else 2
            last_pos = pos
    
    # Add the remaining part from the current parent
    if current_parent == 1:
        result += chromosome1[last_pos:]
    else:
        result += chromosome2[last_pos:] if last_pos < len(chromosome2) else ""
    
    # Ensure the result doesn't exceed maximum length
    if len(result) > MAX_CHROMOSOME_LENGTH:
        result = result[:MAX_CHROMOSOME_LENGTH]
    
    return result

def create_offspring(parent1: Agent, parent2: Agent, llm_interface=None) -> Agent:
    # Create new agent from parent chromosomes
    new_agent = Agent()
    
    # Always combine merging chromosomes using standard method
    merging_instruction = combine_chromosomes(parent1, parent2, "merging")
    new_agent.chromosomes["merging"] = merging_instruction
    
    # Use LLM-based combination if available
    if llm_interface and parent1.chromosomes.get("task") and parent2.chromosomes.get("task"):
        # Use LLM to combine task chromosomes based on merging instructions
        from .constants import TEST_TOKEN_LIMIT
        new_agent.chromosomes["task"] = llm_interface.combine_chromosomes_with_llm(
            parent1.chromosomes["task"],
            parent2.chromosomes["task"],
            merging_instruction,
            max_tokens=TEST_TOKEN_LIMIT
        )
    else:
        # Fallback to standard combination if LLM not available
        new_agent.chromosomes["task"] = combine_chromosomes(parent1, parent2, "task")
    
    # TODO: Add more chromosome types for different aspects of agent behavior
    # TODO: Implement more sophisticated crossover strategies
    # TODO: Add mutation rate as an evolvable parameter
    
    return new_agent

# External command evaluation moved to evaluation.py
