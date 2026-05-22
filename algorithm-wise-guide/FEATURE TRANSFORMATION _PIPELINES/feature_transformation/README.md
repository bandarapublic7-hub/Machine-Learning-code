# Feature Transformation

This is my quick note for Feature Transformation.

Transforms reshape raw columns into forms that are easier for models to learn from.

## Main types
- FunctionTransformer-style thinking
- Power transforms
- Binning and binarization
- Mixed variable cleanup
- Date and time feature extraction

## What this topic does
It changes skewed numbers, turns ranges into buckets, breaks timestamps into useful parts, and simplifies mixed-format values.

## Simple real-life example
A calendar date like 2026-05-15 is not useful by itself, but month, weekday, and weekend flag can be very useful.

