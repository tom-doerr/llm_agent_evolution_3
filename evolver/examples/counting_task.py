#!/usr/bin/env python3
# Example script for the counting task.
# This script takes agent output as input and returns a score.
#
# The hidden goal is to maximize the number of 'a' characters up to 
# TEST_OPTIMAL_LENGTH (23) characters, with a penalty for each character beyond that.

import sys
from evolver.constants import TEST_OPTIMAL_LENGTH

# Import the evaluation function from our module
from evolver.evaluation import evaluate_agent_output

def evaluate_output(text):
    # Use the consolidated evaluation function
    return evaluate_agent_output(text, TEST_OPTIMAL_LENGTH)

if __name__ == "__main__":
    # Read input from stdin
    agent_output = sys.stdin.read().strip()
    
    # Evaluate the output
    score = evaluate_output(agent_output)
    
    # Print some information about the evaluation
    print(f"Input length: {len(agent_output)}")
    print(f"Number of 'a's: {agent_output.count('a')}")
    print(f"Optimal length: {TEST_OPTIMAL_LENGTH}")
    print(f"Penalty: {max(0, len(agent_output) - TEST_OPTIMAL_LENGTH)}")
    print(f"Score calculation: {agent_output.count('a')} - {max(0, len(agent_output) - TEST_OPTIMAL_LENGTH)}")
    
    # The last line must be the score (as a float)
    print(f"{score}")
