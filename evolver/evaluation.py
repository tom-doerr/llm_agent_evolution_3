import subprocess
from typing import Optional

from .agent import Agent
from .constants import TEST_OPTIMAL_LENGTH

def evaluate_agent_output(output: str, optimal_length: int = TEST_OPTIMAL_LENGTH) -> float:
    """Evaluate agent output based on the counting task criteria."""
    count_a = output.count('a')
    penalty = max(0, len(output) - optimal_length)
    return count_a - penalty

def external_command_evaluation(agent: Agent, command: str) -> float:
    """Run external command for evaluation and return the score."""
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
            print(f"Command failed with exit code {process.returncode}")
            if stderr:
                print(f"Error: {stderr.strip()}")
            return 0.0
        
        # Extract score from the last line of output
        lines = stdout.strip().split('\n')
        if not lines:
            return 0.0
            
        try:
            score = float(lines[-1])
            return score
        except ValueError:
            print(f"Invalid score format in command output: {lines[-1]}")
            return 0.0
            
    except (subprocess.SubprocessError, OSError) as error:
        print(f"Error running external command: {error}")
        return 0.0
    except Exception as error:
        print(f"Unexpected error in external evaluation: {error}")
        return 0.0

def evaluate_agent(agent: Agent, eval_command: str = "", optimal_length: int = TEST_OPTIMAL_LENGTH) -> float:
    """Evaluate an agent using either external command or default logic."""
    if eval_command:
        return external_command_evaluation(agent, eval_command)
    
    # Default evaluation using the agent's task chromosome as output
    return evaluate_agent_output(agent.chromosomes["task"], optimal_length)
import sys
import subprocess
from typing import Optional

from .agent import Agent
from .constants import TEST_OPTIMAL_LENGTH

def evaluate_agent_output(output: str, optimal_length: int = TEST_OPTIMAL_LENGTH) -> float:
    """
    Evaluate agent output based on the counting task criteria.
    Rewards 'a' characters up to optimal_length, penalizes excess characters.
    """
    count_a = output.count('a')
    penalty = max(0, len(output) - optimal_length)
    return count_a - penalty

def external_command_evaluation(agent: Agent, command: str) -> float:
    """
    Evaluate an agent using an external command.
    The agent's task chromosome is used as input to the command.
    The command should return a score as the last line of output.
    """
    try:
        # Run the command and pass the agent's task chromosome as input
        process = subprocess.Popen(
            command.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the agent's task chromosome as input
        stdout, stderr = process.communicate(input=agent.chromosomes["task"])
        
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
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
    """
    Evaluate an agent using either external command or default logic.
    """
    if eval_command:
        return external_command_evaluation(agent, eval_command)
    
    # Default evaluation using the agent's task chromosome as output
    return evaluate_agent_output(agent.chromosomes["task"], optimal_length)
