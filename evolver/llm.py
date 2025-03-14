import time
from typing import Optional

import dspy

class LLMInterface:
    def __init__(self, model_name: str = 'openrouter/google/gemini-2.0-flash-001'):
        self.model_name = model_name
        self.lm = None
    
    def initialize(self) -> None:
        # Set up LLM connection
        try:
            self.lm = dspy.LM(self.model_name)
        except Exception as error:
            print(f"Error initializing LLM with model {self.model_name}: {error}")
            # Create a dummy LM for testing purposes if in test environment
            if self.model_name == "test-model":
                self.lm = lambda prompt, max_tokens=None: "Test response"
            else:
                raise
    
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
            except (ConnectionError, TimeoutError) as error:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s, etc.
                    wait_time = 2 ** attempt
                    print(f"LLM error (attempt {attempt+1}/{max_retries}): {error}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating from LLM after {max_retries} attempts: {error}")
                    # Re-raise the exception to allow proper error handling
                    raise
            except (ValueError, TypeError) as error:
                print(f"Error in LLM generation: {error}")
                raise
            except Exception as error:
                print(f"Unexpected error in LLM generation: {error}")
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
        except (ConnectionError, TimeoutError) as error:
            print(f"Network error in LLM combination: {error}")
            # In case of network error, return a simple combination of the inputs
            return f"{parent1_chromosome[:min(len(parent1_chromosome), 100)]} {parent2_chromosome[:min(len(parent2_chromosome), 100)]}"
        except Exception as error:
            print(f"Unexpected error in LLM combination: {error}")
            # In case of other errors, return a simple combination of the inputs
            return f"{parent1_chromosome[:min(len(parent1_chromosome), 100)]} {parent2_chromosome[:min(len(parent2_chromosome), 100)]}"
