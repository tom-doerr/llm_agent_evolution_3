import dspy
from typing import Optional, Dict, Any

class LLMInterface:
    def __init__(self, model_name: str = 'openrouter/google/gemini-2.0-flash-001'):
        self.model_name = model_name
        self.lm = None
    
    def initialize(self) -> None:
        # Set up LLM connection
        self.lm = dspy.LM(self.model_name)
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        # Generate text based on prompt
        if not self.lm:
            self.initialize()
        
        # Try up to 3 times with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.lm(prompt, max_tokens=max_tokens)
                
                # Handle response which might be a list or string
                if isinstance(response, list):
                    response = response[0] if response else ""
                
                return str(response)
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s, etc.
                    import time
                    wait_time = 2 ** attempt
                    print(f"LLM error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating from LLM after {max_retries} attempts: {e}")
                    # Re-raise the exception to allow proper error handling
                    raise
    
    def combine_chromosomes_with_llm(self, parent1_chromosome: str, parent2_chromosome: str, 
                                     instruction_chromosome: str, max_tokens: Optional[int] = None) -> str:
        # Use LLM to combine chromosomes based on instruction
        if not parent1_chromosome and not parent2_chromosome:
            return ""
        
        if not parent1_chromosome:
            return parent2_chromosome
        
        if not parent2_chromosome:
            return parent1_chromosome
        
        # Default instruction if none provided
        if not instruction_chromosome:
            instruction_chromosome = "Combine these two inputs in a creative way."
        
        prompt = f"""
{instruction_chromosome}

Input 1:
{parent1_chromosome}

Input 2:
{parent2_chromosome}
"""
        
        try:
            result = self.generate(prompt, max_tokens=max_tokens)
            return result
        except Exception as e:
            print(f"Error in LLM combination: {e}")
            # In case of error, return a simple combination of the inputs
            # Make sure we include parts of both parents
            return f"{parent1_chromosome[:min(len(parent1_chromosome), 100)]} {parent2_chromosome[:min(len(parent2_chromosome), 100)]}"
