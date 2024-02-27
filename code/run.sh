#!/bin/bash

source activate ExperimentTwo
for ((i=1; i<=20; i++)); do
    cd "$(dirname "$0")"
    python CleanRL.py --t_weight $((i))
    sleep 5s
done
