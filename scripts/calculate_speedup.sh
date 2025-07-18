#!/bin/bash

fileA="$1"
fileB="$2"

if [[ -z "$fileA" || -z "$fileB" ]]; then
    echo "Usage: $0 fileA.csv fileB.csv"
    exit 1
fi

# Output header
echo "tensor, speedup"

# Read both files, skipping headers, line by line
paste -d, <(tail -n +2 "$fileA") <(tail -n +2 "$fileB") | \
while IFS=, read -r tensorA timeA tensorB timeB; do
    if [[ "$tensorA" == "$tensorB" && -n "$timeA" && -n "$timeB" ]]; then
        speedup=$(awk -v a="$timeA" -v b="$timeB" 'BEGIN { printf "%.6f", a / b }')
        echo "$tensorA, $speedup"
    else
        echo "Mismatch or missing data: $tensorA vs $tensorB" >&2
    fi
done
