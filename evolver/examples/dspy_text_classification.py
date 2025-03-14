#!/usr/bin/env python3
# Example script to run the DSPyOptimizer on a text classification task

import os
import sys
import dspy

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evolver.dspy_optimizer import DSPyOptimizer

# Create a simple text classifier
class TextClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prompt = "Classify the text as positive or negative."
    
    def forward(self, text):
        # Use the prompt to classify the text
        full_prompt = f"{self.prompt}\n\nText: {text}\n\nClassification:"
        response = dspy.LM("openrouter/google/gemini-2.0-flash-001")(full_prompt, max_tokens=10)
        
        # Handle response which might be a list or string
        if isinstance(response, list):
            response = response[0] if response else ""
        
        # Clean up response
        response = str(response).strip().lower()
        if "positive" in response:
            return "positive"
        if "negative" in response:
            return "negative"
        return response

# Define evaluation metric
def metric(result, example):
    # Simple accuracy metric
    if result == example["label"]:
        return 1.0
    return 0.0

def run_optimization(module, metric, trainset, testset=None, max_evaluations=20):
    # Create optimizer
    optimizer = DSPyOptimizer(
        max_agents=20,  # Small population for demo
        parallel=2,     # Few parallel agents for demo
        verbose=True
    )
    
    print("Starting optimization...")
    
    # Optimize module
    optimized_module = optimizer.optimize(
        module=module,
        metric=metric,
        trainset=trainset,
        max_evaluations=max_evaluations
    )
    
    # Evaluate on test set if provided
    if testset:
        print("\nEvaluating on test set:")
        correct = 0
        for example in testset:
            prediction = optimized_module(example["text"])
            is_correct = prediction == example["label"]
            correct += int(is_correct)
            print(f"Text: {example['text']}")
            print(f"Prediction: {prediction}, Actual: {example['label']}, Correct: {is_correct}")
        
        print(f"\nTest accuracy: {correct / len(testset) * 100:.1f}%")
    
    print(f"Optimized prompt: {optimized_module.prompt}")
    return optimized_module

def main():
    # Create simple training data
    trainset = [
        {"text": "I love this product, it's amazing!", "label": "positive"},
        {"text": "This is the worst experience ever.", "label": "negative"},
        {"text": "The service was excellent and the staff was friendly.", "label": "positive"},
        {"text": "I'm very disappointed with the quality.", "label": "negative"},
        {"text": "Highly recommended, would buy again.", "label": "positive"}
    ]
    
    # Create test data
    testset = [
        {"text": "I'm really happy with my purchase.", "label": "positive"},
        {"text": "Don't waste your money on this.", "label": "negative"}
    ]
    
    # Create module
    module = TextClassifier()
    
    # Run optimization
    run_optimization(module, metric, trainset, testset)

if __name__ == "__main__":
    main()
