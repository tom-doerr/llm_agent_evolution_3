import argparse
from typing import Dict, Any

def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evolutionary algorithm with LLM-based components")
    
    # Core parameters
    parser.add_argument("--parallel", type=int, default=10,
                        help="Number of agents to run in parallel")
    parser.add_argument("--max-agents", type=int, default=1_000_000,
                        help="Maximum population size")
    
    # I/O options
    parser.add_argument("--save", type=str, default="",
                        help="Save best agent to file")
    parser.add_argument("--load", type=str, default="",
                        help="Load agent from file")
    
    # Evaluation options
    parser.add_argument("--eval-command", type=str, default="",
                        help="Command for evaluation")
    
    # Output options
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    # LLM options
    parser.add_argument("--model", type=str, default="openrouter/google/gemini-2.0-flash-001",
                        help="LLM model to use")
    
    return vars(parser.parse_args())
