#!/bin/bash

# Set default parameters
EPSILON=1.0
DELTA=1e-5
SAMPLES=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epsilon)
            EPSILON="$2"
            shift 2
            ;;
        --delta)
            DELTA="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print the parameters being used
echo "Running Gaussian mechanism with:"
echo "Epsilon: $EPSILON"
echo "Delta: $DELTA"
echo "Samples: $SAMPLES"

# Run the Gaussian mechanism
python gaussian_mechanism.py \
    --data data/heart_failure_clinical_records_dataset.csv \
    --epsilon $EPSILON \
    --delta $DELTA \
    --samples $SAMPLES 