# Column Transformers and Pipelines

This is my quick note for Column Transformers and Pipelines.

Different columns need different preprocessing, and a pipeline keeps the full workflow in one reliable object.

## Main types
- ColumnTransformer
- Numeric preprocessing
- Categorical preprocessing
- End-to-end pipeline
- Feature name tracking

## What this topic does
It applies the right transformer to each column group and then pushes the cleaned result into a model.

## Simple real-life example
In a kitchen, vegetables are chopped, rice is washed, and spices are mixed differently before everything goes into one dish. Pipelines do the same with features.

