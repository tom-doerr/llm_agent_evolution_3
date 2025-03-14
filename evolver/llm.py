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
        
        try:
            response = self.lm(prompt, max_tokens=max_tokens)
            return response
        except Exception as e:
            print(f"Error generating from LLM: {e}")
            return ""
    
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
        
        return self.generate(prompt, max_tokens=max_tokens)
