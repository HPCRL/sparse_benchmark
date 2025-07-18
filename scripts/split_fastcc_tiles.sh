#!/bin/bash

input="${1:-/dev/stdin}"  # Input file or stdin
prefix="${2:-split}"      # Optional output prefix

match_file="${prefix}_molecules.csv"
other_file="${prefix}_frostt.csv"

# Empty output files and write headers
echo "tensor, tile_size, time" > "$match_file"
echo "tensor, tile_size, time" > "$other_file"

tail -n +2 "$input" | while IFS= read -r line; do
    tensor=$(echo "$line" | cut -d',' -f1 | xargs)
    if [[ "$tensor" == caffeine* || "$tensor" == guanine* ]]; then
        echo "$line" >> "$match_file"
    else
        echo "$line" >> "$other_file"
    fi
done

echo "Wrote:"
echo " - Matching:    $match_file"
echo " - Non-matching: $other_file"