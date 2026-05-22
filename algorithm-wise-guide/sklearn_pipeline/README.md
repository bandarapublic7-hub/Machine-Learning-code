# Sklearn Pipeline

This is my quick note for Sklearn Pipeline.

A pipeline keeps the preprocessing steps and the model together so the same flow runs every time.

## Main types
- Preprocess plus model
- Reusable end-to-end workflow

## What this topic does
It helps avoid mistakes by applying the same training transformations during prediction time.

## Simple real-life example
If you clean, encode, and scale data by hand every time, it is easy to miss one step. Pipeline locks the sequence.

