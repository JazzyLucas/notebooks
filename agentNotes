# Building Custom AI Agents: A Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Required Imports](#required-imports)
3. [Resource Management](#resource-management)
4. [Model Setup](#model-setup)
5. [Custom Tools](#custom-tools)
6. [Agent Creation](#agent-creation)
7. [Best Practices](#best-practices)
8. [Example Applications](#example-applications)
9. [Advanced Features](#advanced-features)

## Overview

This guide details how to create custom AI agents using the `smolagents` library and Transformer models. It includes complete code examples, best practices, and implementations for various use cases.

## Required Imports

```python
from smolagents import CodeAgent, TransformersModel, Tool
import torch
import signal
import atexit
import gc
import re
```

## Resource Management

### GPU Memory Manager
```python
class GPUMemoryManager:
    """Context manager for GPU memory management"""
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available memory
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_gpu_memory()

def cleanup_gpu_memory():
    """Clean up GPU memory and release resources"""
    print("Cleaning up GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.cuda.device('cuda'):
            torch.cuda.ipc_collect()
    gc.collect()
    print("Cleanup complete")

# Register cleanup for program exit
atexit.register(cleanup_gpu_memory)

# Handle termination signals
def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}")
    cleanup_gpu_memory()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

## Model Setup

```python
def initialize_model():
    return TransformersModel(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device_map="auto",
        max_new_tokens=512,
        torch_dtype=torch.float16,
    )
```

## Custom Tools

### Base Tool Template
```python
class CustomTool(Tool):
    name = "tool_name"
    description = "What your tool does"
    inputs = {
        "parameter_name": {
            "type": "data_type",
            "description": "Parameter description"
        }
    }
    output_type = "expected_output_type"

    def forward(self, *args):
        # Implementation here
        pass
```

### Text Analysis Tool
```python
class TextAnalysisTool(Tool):
    name = "analyze_text"
    description = "Analyzes text for sentiment and key metrics"
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to analyze"
        }
    }
    output_type = "dict"

    def forward(self, text):
        results = {
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(text.split('.')),
            "average_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text else 0
        }
        return results
```

### Data Analysis Tool
```python
class DataAnalysisTool(Tool):
    name = "analyze_data"
    description = "Performs statistical analysis on numerical data"
    inputs = {
        "data": {
            "type": "list",
            "description": "List of numbers to analyze"
        },
        "metrics": {
            "type": "list",
            "description": "List of metrics to calculate"
        }
    }
    output_type = "dict"

    def forward(self, data, metrics):
        results = {}
        if "mean" in metrics:
            results["mean"] = sum(data) / len(data)
        if "max" in metrics:
            results["max"] = max(data)
        if "min" in metrics:
            results["min"] = min(data)
        if "range" in metrics:
            results["range"] = max(data) - min(data)
        return results
```

## Agent Creation

### Basic Agent Setup
```python
def create_custom_agent(model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", tools=None):
    with GPUMemoryManager():
        model = initialize_model()
        
        if tools is None:
            tools = [TextAnalysisTool(), DataAnalysisTool()]
        
        agent = CodeAgent(
            tools=tools,
            model=model,
            add_base_tools=False
        )
        
        return agent
```

## Best Practices

### 1. Error Handling
```python
def safe_execution(func):
    """Decorator for safe function execution"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error executing {func.__name__}: {e}")
            return None
    return wrapper
```

### 2. Memory Management
```python
class MemoryAwareTool(Tool):
    def __init__(self):
        self.memory = {}
        self._max_memory_size = 1000
    
    def store_result(self, key, value):
        if len(self.memory) >= self._max_memory_size:
            # Remove oldest entry
            oldest_key = next(iter(self.memory))
            del self.memory[oldest_key]
        self.memory[key] = value
    
    def get_result(self, key):
        return self.memory.get(key)
    
    def clear_memory(self):
        self.memory.clear()
```

## Example Applications

### 1. Text Processing Agent
```python
class TextProcessor:
    def __init__(self):
        self.agent = create_custom_agent(tools=[TextAnalysisTool()])
    
    def process_document(self, text):
        prompt = f"""Analyze the following text and provide key metrics:
        
        Text: {text}
        
        Please provide:
        1. Word count
        2. Sentence count
        3. Average word length
        """
        return self.agent.run(prompt)
```

### 2. Data Analysis Agent
```python
class DataAnalyzer:
    def __init__(self):
        self.agent = create_custom_agent(tools=[DataAnalysisTool()])
    
    def analyze_dataset(self, data, metrics=None):
        if metrics is None:
            metrics = ["mean", "max", "min", "range"]
            
        prompt = f"""Analyze the following dataset using these metrics: {metrics}
        
        Data: {data}
        
        Provide a comprehensive analysis with all requested metrics.
        """
        return self.agent.run(prompt)
```

## Advanced Features

### 1. Chain of Thought Reasoning
```python
class ReasoningTool(Tool):
    name = "reasoning_tool"
    description = "Performs step-by-step reasoning"
    inputs = {
        "problem": {
            "type": "string",
            "description": "Problem to analyze"
        }
    }
    output_type = "dict"

    def forward(self, problem):
        steps = []
        
        # Step 1: Problem Analysis
        steps.append({
            "step": "analysis",
            "description": "Analyzing problem structure",
            "result": self.analyze_problem(problem)
        })
        
        # Step 2: Solution Planning
        steps.append({
            "step": "planning",
            "description": "Planning solution approach",
            "result": self.plan_solution(problem)
        })
        
        # Step 3: Implementation
        steps.append({
            "step": "implementation",
            "description": "Implementing solution",
            "result": self.implement_solution(problem)
        })
        
        return {
            "steps": steps,
            "conclusion": self.conclude(steps)
        }
    
    def analyze_problem(self, problem):
        # Implementation specific to your use case
        pass
    
    def plan_solution(self, problem):
        # Implementation specific to your use case
        pass
    
    def implement_solution(self, problem):
        # Implementation specific to your use case
        pass
    
    def conclude(self, steps):
        # Implementation specific to your use case
        pass
```

### 2. Interactive Agent
```python
class InteractiveAgent:
    def __init__(self):
        self.agent = create_custom_agent()
        self.conversation_history = []
    
    def interact(self, user_input):
        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Create context from history
        context = "\n".join([f"{msg['role']}: {msg['content']}" 
                           for msg in self.conversation_history])
        
        # Get agent response
        response = self.agent.run(f"""
        Previous conversation:
        {context}
        
        Please provide a response to the latest user input.
        """)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        self.conversation_history.clear()
```

## Usage Example

```python
def main():
    # Create and initialize agents
    text_processor = TextProcessor()
    data_analyzer = DataAnalyzer()
    interactive_agent = InteractiveAgent()
    
    # Example text processing
    text_result = text_processor.process_document(
        "This is a sample text. It contains multiple sentences. Each one is unique."
    )
    print("Text Analysis Result:", text_result)
    
    # Example data analysis
    data_result = data_analyzer.analyze_dataset(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    print("Data Analysis Result:", data_result)
    
    # Interactive session
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = interactive_agent.interact(user_input)
        print("Agent:", response)

if __name__ == "__main__":
    main()
```

This guide provides a comprehensive foundation for building custom AI agents. You can extend and modify these components based on your specific needs. Remember to always implement proper error handling, resource management, and cleanup procedures in your agent implementations.

Key things to remember:
10. Always manage GPU memory properly
11. Implement proper error handling
12. Use context managers for resource management
13. Document your code thoroughly
14. Test your agents with various inputs
15. Monitor performance and memory usage
16. Implement proper cleanup procedures

For more advanced applications, consider implementing:
- Asynchronous processing
- Distributed computing capabilities
- Custom model loading and management
- Advanced memory management strategies
- Specialized tools for your domain


# Basic Agent example 1
```python
# Required imports for Transformers and smolagents

from smolagents import CodeAgent, TransformersModel

  

# Initialize the model

model = TransformersModel(

model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", # This is the default model for smolagents

device_map="auto", # Use available device (CPU/GPU)

max_new_tokens=1024, # Maximum number of new tokens to generate

)

  

# Create the agent with the model

agent = CodeAgent(tools=[], model=model, add_base_tools=True)

  

# Run the agent with our Fibonacci question

response = agent.run(

"Calculate the 118th number in the Fibonacci sequence. Show your work step by step.",

)

  

print("\nFinal Response:", response)
```

### requirements.txt
```txt
smolagents>=0.1.0

transformers>=4.36.0

torch>=2.1.0
```
# Basic Agent example 2
```python
# Required imports for Transformers and smolagents

from smolagents import CodeAgent, TransformersModel, Tool

import re

import torch

import signal

import atexit

import gc

  

# Track if cleanup has already been performed

_cleanup_performed = False

  

def get_ordinal(n):

"""Convert a number to its ordinal representation (1st, 2nd, 3rd, etc.)"""

if 10 <= n % 100 <= 20:

suffix = 'th'

else:

suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

return f"{n}{suffix}"

  

def cleanup_gpu_memory():

"""Clean up GPU memory and release resources"""

global _cleanup_performed

if _cleanup_performed:

return

print("Cleaning up GPU memory...")

if torch.cuda.is_available():

torch.cuda.empty_cache()

torch.cuda.synchronize()

# Clear CUDA memory cache

with torch.cuda.device('cuda'):

torch.cuda.ipc_collect()

gc.collect() # Run garbage collection

print("Cleanup complete")

_cleanup_performed = True

  

# Register cleanup function to run on exit

atexit.register(cleanup_gpu_memory)

  

# Handle termination signals

def signal_handler(signum, frame):

print(f"\nReceived signal {signum}")

cleanup_gpu_memory()

exit(0)

  

signal.signal(signal.SIGINT, signal_handler)

signal.signal(signal.SIGTERM, signal_handler)

  

# Get the Fibonacci number from user input

while True:

try:

n = int(input("Enter the Fibonacci number to calculate (e.g., 13): "))

if n < 0:

print("Please enter a non-negative number.")

continue

break

except ValueError:

print("Please enter a valid integer.")

  

# Free any existing GPU memory

cleanup_gpu_memory()

  

class GPUMemoryManager:

"""Context manager for GPU memory management"""

def __enter__(self):

if torch.cuda.is_available():

torch.cuda.empty_cache()

# Set memory allocator settings

torch.cuda.set_per_process_memory_fraction(0.7) # Use only 70% of available memory

return self

  

def __exit__(self, exc_type, exc_val, exc_tb):

cleanup_gpu_memory()

  

# Initialize the model with memory optimizations

with GPUMemoryManager():

model = TransformersModel(

model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", # This is the default model for smolagents

device_map="auto", # Use available device (CPU/GPU)

max_new_tokens=512, # Reduce max tokens to save memory

torch_dtype=torch.float16, # Use half precision to reduce memory usage

)

  

# Create a tool to handle code execution

class FibonacciTool(Tool):

name = "execute_fibonacci"

description = "Execute Python code to calculate a Fibonacci number. You must define a function that calculates the nth Fibonacci number."

inputs = {

"code": {

"type": "string",

"description": "The Python code that defines and calls a Fibonacci function. The code must define a function that takes n as input and returns the nth Fibonacci number."

}

}

output_type = "integer"

  

def __init__(self, target_n):

super().__init__()

self.has_executed = False

self.result = None

self.target_n = target_n

self.ordinal = get_ordinal(target_n)

  

def forward(self, code):

# Only execute once

if self.has_executed:

return self.result

  

try:

# Create a clean namespace for execution

namespace = {}

# Execute the code

exec(code, namespace)

# Look for a function call that returns a number

for name, value in namespace.items():

if callable(value) and name.lower().startswith('fib'):

# Found a Fibonacci function, try to get the result

try:

self.result = value(self.target_n) # Call with user's n

if isinstance(self.result, (int, float)):

self.has_executed = True

print(f"The {self.ordinal} Fibonacci number is: {self.result}")

return self.result

except:

continue

return None

except Exception as e:

print(f"Error executing code: {e}")

return None

  

# Create a tool to show the final answer

class FinalAnswerTool(Tool):

name = "final_answer"

description = "Show the final answer for the Fibonacci calculation"

inputs = {

"result": {

"type": "integer",

"description": "The calculated Fibonacci number"

}

}

output_type = "string"

  

def forward(self, result):

print(f"\nFinal Answer: {result}")

return str(result)

  

# Create the agent with our custom tools

agent = CodeAgent(

tools=[FibonacciTool(n), FinalAnswerTool()],

model=model,

add_base_tools=False

)

  

try:

# Run the agent with our Fibonacci question

response = agent.run(

f"""Write ONE Python function to calculate the nth number in the Fibonacci sequence, then use it to find the {get_ordinal(n)} number.

Requirements:

1. Define EXACTLY ONE function named 'fibonacci' that takes n as input and returns the nth Fibonacci number

2. The function should handle n=0 (return 0) and n=1 (return 1)

3. For n>1, calculate F(n) = F(n-1) + F(n-2)

4. Call the function with n={n} and use the final_answer tool to display the result

5. DO NOT define multiple versions of the function - choose ONE implementation method

Choose ONE method to implement (iterative, recursive, or closed-form) and explain your solution in a brief comment."""

)

print("\nFinal Response:", response)

except Exception as e:

print(f"Error occurred: {e}")

raise
```

### requirements.txt
```txt
transformers>=4.36.0

torch>=2.1.0

smolagents>=0.1.0
```
