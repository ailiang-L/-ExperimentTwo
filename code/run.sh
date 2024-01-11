#!/bin/bash

source activate ExperimentTwo
for ((i=1; i<=100; i++)); do
    cd "$(dirname "$0")"
    python CleanRL.py --e_weight $((i))
    sleep 5s
done
