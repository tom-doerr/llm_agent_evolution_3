#!/bin/bash

# Check if number of lines is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <number_of_lines>"
    exit 1
fi

# Get the number of lines requested
n=$1

# Check if spec.md exists
if [ ! -f "spec.md" ]; then
    echo "Error: spec.md file not found"
    exit 1
fi

# Count total lines in spec.md
total_lines=$(wc -l < spec.md)

# Ensure we don't try to sample more lines than exist
if [ "$n" -gt "$total_lines" ]; then
    echo "Warning: Requested $n lines but spec.md only has $total_lines lines"
    n=$total_lines
fi

# Sample n random lines
echo "Sampling $n random lines from spec.md:"
echo "----------------------------------------"
shuf -n "$n" spec.md
echo "----------------------------------------"
