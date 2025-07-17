#!/bin/bash
input="${1:-/dev/stdin}"  # default to stdin if no file provided

# Output header
echo "tensor, tile_size, time"

# Parse each matching line
grep -E "Time taken for .+: .+ seconds at tile size .+" "$input" | \
while read -r line; do
    tensor=$(echo "$line" | sed -n 's/^Time taken for \([^:]*\):.*/\1/p')
    time=$(echo "$line"   | sed -n 's/^Time taken for [^:]*: \([0-9.]*\) seconds.*/\1/p')
    tile=$(echo "$line"   | sed -n 's/.*tile size \([0-9]*\)$/\1/p')

    if [[ -n "$tensor" && -n "$time" && -n "$tile" ]]; then
        echo "$tensor, $tile, $time"
    fi
done