# Column Transformer

This is my quick note for Column Transformer.

Different columns need different preprocessing, and ColumnTransformer handles that in one place.

## Main types
- Numeric transformation
- Categorical transformation
- Mixed data preprocessing

## What this topic does
It applies the right transformation to the right columns and combines the output into one final feature table.

## Simple real-life example
Age may need scaling, city may need encoding, and missing temperature may need imputation. ColumnTransformer keeps all three together.

