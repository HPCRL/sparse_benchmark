#!/bin/bash

input="${1:-/dev/stdin}"  # Read from file or stdin

# Use awk to do all parsing and formatting
gawk -F',' '
BEGIN {
    OFS = ",";
}

NR > 1 {
    # Clean up input
    gsub(/^ +| +$/, "", $1);  # trim tensor
    gsub(/^ +| +$/, "", $2);  # trim tile size
    gsub(/^ +| +$/, "", $3);  # trim time

    tensor = $1;
    tile = $2;
    time = $3;

    # Canonicalize tensor name (spaces → dashes)
    gsub(/ /, "-", tensor);

    # Record the value
    data[tile, tensor] = time;
    tiles[tile] = 1;
    tensors[tensor] = 1;
}

END {
    # Sort tensor names for column headers
    n = asorti(tensors, tensor_list);
    
    # Sort tiles numerically
    m = 0;
    for (tile in tiles) {
        tile_list[++m] = tile;
    }
    # Simple numeric sort for tiles
    for (i = 1; i < m; i++) {
        for (j = i + 1; j <= m; j++) {
            if (tile_list[i] + 0 > tile_list[j] + 0) {
                temp = tile_list[i];
                tile_list[i] = tile_list[j];
                tile_list[j] = temp;
            }
        }
    }

    # Print header
    printf "tile";
    for (i = 1; i <= n; i++) {
        printf ",%s", tensor_list[i];
    }
    print "";

    # Print rows
    for (i = 1; i <= m; i++) {
        tile = tile_list[i];
        printf "%s", tile;
        for (j = 1; j <= n; j++) {
            t = tensor_list[j];
            key = tile SUBSEP t;
            val = (key in data) ? data[key] : "";
            printf ",%s", val;
        }
        print "";
    }
}
' "$input"
