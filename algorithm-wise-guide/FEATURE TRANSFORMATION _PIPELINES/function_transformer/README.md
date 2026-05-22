# Function Transformer

This is my quick note for Function Transformer.

FunctionTransformer is useful when I want to apply my own small preprocessing logic inside an sklearn flow.

## Main types
- Custom math transforms
- Reusable preprocessing functions

## What this topic does
It wraps a normal Python function so it can behave like an sklearn transformer.

## Simple real-life example
If I want to take log of salary or add 1 before a square root, I can keep that step inside the pipeline.

