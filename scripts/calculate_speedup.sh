#!/bin/bash
#!/bin/bash
fileA="$1"
fileB="$2"

if [[ -z "$fileA" || -z "$fileB" ]]; then
    echo "Usage: $0 fileA.csv fileB.csv"
    exit 1
fi

# Output header
echo "tensor, speedup"

# Skip headers and process both files
join -t, -1 1 -2 1 <(tail -n +2 "$fileA" | sort) <(tail -n +2 "$fileB" | sort) | \
while IFS=, read -r tensor timeA timeB; do
    if [[ -n "$tensor" && -n "$timeA" && -n "$timeB" ]]; then
        speedup=$(awk -v a="$timeA" -v b="$timeB" 'BEGIN { printf "%.6f", a / b }')
        echo "$tensor, $speedup"
    fi
done
