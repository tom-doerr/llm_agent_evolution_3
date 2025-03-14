import subprocess
from typing import Optional

from .agent import Agent
from .constants import TEST_OPTIMAL_LENGTH

def evaluate_agent_output(output: str, optimal_length: int = TEST_OPTIMAL_LENGTH) -> float:
    # Evaluate output based on the counting task criteria
    # Rewards 'a' characters up to optimal_length, penalizes excess characters
    count_a = output.count('a')
    penalty = max(0, len(output) - optimal_length)
    return count_a - penalty

def external_command_evaluation(agent: Agent, command: str) -> float:
    # Run external command for evaluation and return the score
    try:
        # Get agent output based on task chromosome
        agent_output = agent.chromosomes["task"]
        
        # Run the command with agent output as input
        process = subprocess.Popen(
            command.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the agent's task chromosome as input
        stdout, stderr = process.communicate(input=agent_output)
        
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            if stderr:
                print(f"Error: {stderr}")
            return 0.0
        
        # Parse the output to get the score (last line)
        lines = stdout.strip().split('\n')
        if not lines:
            return 0.0
            
        try:
            # The score should be the last line
            return float(lines[-1])
        except ValueError:
            print(f"Could not parse score from output: {lines[-1]}")
            return 0.0
            
    except Exception as e:
        print(f"Error running command: {e}")
        return 0.0

def evaluate_agent(agent: Agent, eval_command: str = "", optimal_length: int = TEST_OPTIMAL_LENGTH) -> float:
    # Evaluate an agent using either external command or default logic
    if eval_command:
        return external_command_evaluation(agent, eval_command)
    
    # Default evaluation using the agent's task chromosome as output
    # TODO: In a more complex implementation, we would generate output using the task chromosome as a prompt
    # For now, we directly evaluate the chromosome content as the output
    output = agent.chromosomes["task"]
    return evaluate_agent_output(output, optimal_length)
