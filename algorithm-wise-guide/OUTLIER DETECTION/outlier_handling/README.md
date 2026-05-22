# Outlier Handling

This is my quick note for Outlier Handling.

Outliers are unusual values, and you should inspect them before deciding whether to keep, clip, or remove them.

## Main types
- Z-score filtering
- IQR filtering
- Percentile capping

## What this topic does
It finds values that sit far away from the rest of the data and shows common rules to deal with them.

## Simple real-life example
If most people commute 5 to 30 km and one row says 5000 km, you should inspect it before training a model.

