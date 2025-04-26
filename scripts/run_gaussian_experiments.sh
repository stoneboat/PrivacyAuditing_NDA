#!/bin/bash

# Default parameters
DELTA=1e-5
SAMPLES=300
OUTPUT_DIR="tmp_data"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Array of epsilon values to test
EPSILON_VALUES=(0.1 1 10)

echo "Starting Gaussian mechanism experiments with different epsilon values..."
echo "Delta: $DELTA"
echo "Samples: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo "Epsilon values to test: ${EPSILON_VALUES[*]}"
echo "----------------------------------------"

# Run experiments for each epsilon value
for EPSILON in "${EPSILON_VALUES[@]}"; do
    echo "Running experiment with epsilon = $EPSILON"
    echo "----------------------------------------"
    
    # Set output file name based on epsilon
    OUTPUT_FILE="$OUTPUT_DIR/private_averages_epsilon_${EPSILON}.csv"
    
    # Run the experiment
    python gaussian_mechanism.py \
        --data data/heart_failure_clinical_records_dataset.csv \
        --epsilon $EPSILON \
        --delta $DELTA \
        --samples $SAMPLES \
        --output $OUTPUT_FILE
    
    echo "----------------------------------------"
    echo "Experiment with epsilon = $EPSILON completed"
    echo "Results saved to: $OUTPUT_FILE"
    echo "----------------------------------------"
done

echo "All experiments completed!"
echo "Results are saved in: $OUTPUT_DIR"
echo "Files created:"
ls -l $OUTPUT_DIR/private_averages_epsilon_*.csv 