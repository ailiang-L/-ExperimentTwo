#!/bin/bash

source activate ExperimentTwo
for ((i=1; i<=100; i++)); do
    cd "$(dirname "$0")"
    python Agent.py $((i+1))
    sleep 5s
done
