import random
import string
import subprocess
from typing import List, Tuple, Optional, Callable
import math

from .agent import Agent
from .constants import MAX_CHROMOSOME_LENGTH

def pareto_weights(scores: List[float]) -> List[float]:
    # Calculate Pareto distribution weights (fitness^2)
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
    
    # Calculate weights as score^2
    return [score * score for score in adjusted_scores]

def select_parents(population: List[Agent], num_parents: int) -> List[Agent]:
    # Select parents using Pareto distribution
    if not population or num_parents <= 0:
        return []
    
    scores = [agent.score for agent in population]
    weights = pareto_weights(scores)
    
    # Ensure num_parents doesn't exceed population size
    num_parents = min(num_parents, len(population))
    
    # Weighted sampling without replacement
    parents = []
    remaining_agents = population.copy()
    remaining_weights = weights.copy()
    
    for _ in range(num_parents):
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
        
        # Add selected agent to parents
        parents.append(remaining_agents[idx])
        
        # Remove selected agent from remaining options
        remaining_agents.pop(idx)
        remaining_weights.pop(idx)
    
    return parents

def find_hotspots(text: str) -> List[int]:
    # Identify punctuation and space positions as potential crossover points
    hotspots = []
    
    # Add punctuation positions
    for i, char in enumerate(text):
        if char in string.punctuation:
            hotspots.append(i)
    
    # Add some space positions with a certain probability
    space_hotspot_probability = 0.3
    for i, char in enumerate(text):
        if char == ' ' and random.random() < space_hotspot_probability:
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
    hotspots = find_hotspots(chromosome1)
    
    # If no hotspots found, create some arbitrary ones
    if not hotspots:
        # Create crossover points roughly every 5-10 characters
        avg_segment_length = random.randint(5, 10)
        for i in range(avg_segment_length, len(chromosome1), avg_segment_length):
            if i < len(chromosome1):
                hotspots.append(i)
    
    # Calculate crossover probability to achieve on average one switch per chromosome
    crossover_probability = 1.0 / (len(hotspots) + 1) if hotspots else 0.5
    
    # Perform crossover
    result = ""
    current_parent = 1  # Start with parent1
    last_pos = 0
    
    for pos in hotspots:
        if random.random() < crossover_probability:
            # Switch parents at this hotspot
            if current_parent == 1:
                result += chromosome1[last_pos:pos]
                current_parent = 2
            else:
                result += chromosome2[last_pos:pos]
                current_parent = 1
            last_pos = pos
    
    # Add the remaining part from the current parent
    if current_parent == 1:
        result += chromosome1[last_pos:]
    else:
        result += chromosome2[last_pos:]
    
    # Ensure the result doesn't exceed maximum length
    if len(result) > MAX_CHROMOSOME_LENGTH:
        result = result[:MAX_CHROMOSOME_LENGTH]
    
    return result

def create_offspring(parent1: Agent, parent2: Agent, llm_interface=None) -> Agent:
    # Create new agent from parent chromosomes
    new_agent = Agent()
    
    # Combine chromosomes
    for chromosome_name in ["task", "merging"]:
        new_agent.chromosomes[chromosome_name] = combine_chromosomes(
            parent1, parent2, chromosome_name
        )
    
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
        
        # Extract score from the last line of output
        lines = stdout.strip().split('\n')
        if lines:
            try:
                score = float(lines[-1])
                return score
            except ValueError:
                # If last line isn't a valid float, return 0
                return 0.0
        
        return 0.0
    except Exception as e:
        print(f"Error in external evaluation: {e}")
        return 0.0
