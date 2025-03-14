#!/usr/bin/env python3
"""
Example script for the counting task.
This script takes agent output as input and returns a score.

The hidden goal is to maximize the number of 'a' characters up to 
TEST_OPTIMAL_LENGTH (23) characters, with a penalty for each character beyond that.
"""

import sys
from evolver.constants import TEST_OPTIMAL_LENGTH

def evaluate_output(text):
    """
    Evaluate the agent output:
    - Count the number of 'a' characters (reward)
    - Apply penalty for length beyond optimal
    """
    count_a = text.count('a')
    penalty = max(0, len(text) - TEST_OPTIMAL_LENGTH)
    score = count_a - penalty
    return score

if __name__ == "__main__":
    # Read input from stdin
    agent_output = sys.stdin.read().strip()
    
    # Evaluate the output
    score = evaluate_output(agent_output)
    
    # Print some information about the evaluation
    print(f"Input length: {len(agent_output)}")
    print(f"Number of 'a's: {agent_output.count('a')}")
    
    # The last line must be the score (as a float)
    print(f"{score}")
