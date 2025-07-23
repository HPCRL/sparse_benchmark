#!/bin/bash

input="${1:-/dev/stdin}"  # CSV file input (from previous script or file)
prefix="${2:-output}"     # Optional prefix for split files

before_file="sparta_frostt.csv"
after_file="sparta_molecules.csv"

found=0

# Empty the output files if they exist
> "$before_file"
> "$after_file"

echo "tensor, time" >> "$before_file"
echo "tensor, time" >> "$after_file"

while IFS= read -r line; do
    if [[ $found -eq 0 && $line == *"caffeine-vvoo"* ]]; then
        found=1
    fi

    if [[ $found -eq 0 ]]; then
        echo "$line" >> "$before_file"
    else
        echo "$line" >> "$after_file"
    fi
done < "$input"

echo "Split complete:"
echo " - Before: $before_file"
echo " - After:  $after_file"