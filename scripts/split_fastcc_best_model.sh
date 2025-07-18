#!/bin/bash

input="$1"

if [[ -z "$input" ]]; then
    echo "Usage: $0 input.csv"
    exit 1
fi

base="${input%.csv}"
best_file="${base}_best.csv"
model_file="${base}_model.csv"

# Write headers
echo "tensor, time" > "$best_file"
echo "tensor, time" > "$model_file"

# Read input line by line
while IFS=, read -r name best model; do
    # Skip header line if present
    if [[ "$name" == "tensor" ]]; then
        continue
    fi

    # Trim whitespace
    name=$(echo "$name" | xargs)
    best=$(echo "$best" | xargs)
    model=$(echo "$model" | xargs)

    echo "$name, $best" >> "$best_file"
    echo "$name, $model" >> "$model_file"
done < "$input"

echo "Wrote:"
echo " - $best_file"
echo " - $model_file"