import string
import subprocess
from typing import List, Tuple, Optional, TypeVar

from .agent import Agent
from .constants import MAX_CHROMOSOME_LENGTH
from .utils import weighted_sample, prepare_weights, create_parent_pairs

T = TypeVar('T')


def select_parents(population: List[Agent], num_parents: int) -> List[Agent]:
    # Select parents using Pareto distribution
    if not population or num_parents <= 0:
        return []
    
    scores = [agent.score for agent in population]
    weights = prepare_weights(scores)
    
    # Use common weighted sampling utility
    return weighted_sample(population, weights, num_parents)

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
    
    return new_agent

def external_command_evaluation(agent: Agent, command: str) -> float:
    # Run external command for evaluation
    try:
        # Get agent output based on task chromosome
        agent_output = agent.chromosomes["task"]
        
        # Run the command with agent output as input
        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=agent_output)
        
        # Check for process errors
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}: {stderr}")
            return 0.0
        
        # Extract score from the last line of output
        return _parse_command_output(stdout)
    except subprocess.SubprocessError as error:
        print(f"Subprocess error in external evaluation: {error}")
        return 0.0
    except Exception as error:
        print(f"Unexpected error in external evaluation: {error}")
        return 0.0

def _parse_command_output(stdout: str) -> float:
    """Parse the command output to extract the score."""
    lines = stdout.strip().split('\n')
    if not lines:
        return 0.0
        
    try:
        score = float(lines[-1])
        return score
    except ValueError:
        # If last line isn't a valid float, return 0
        print(f"Invalid score format in command output: {lines[-1]}")
        return 0.0
