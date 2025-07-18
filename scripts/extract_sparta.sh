#!/bin/bash
input="${1:-/dev/stdin}"  # use stdin if no argument provided

# Initialize state
tensor=""
total=""

# Read line by line
while IFS= read -r line; do
    # Extract tensor name from "Tensor: ..."
    if [[ $line == Tensor:* ]]; then
        tensor=$(echo "$line" | cut -d':' -f2- | xargs)
    fi

    # Extract total time from "[Total time]"
    if [[ $line == \[Total\ time\]* ]]; then
        total=$(echo "$line" | awk '{print $3}')
        
        # Once we have both, print and reset
        if [[ -n $tensor && -n $total ]]; then
            echo "$tensor, $total"
            tensor=""
            total=""
        fi
    fi
done < "$input"