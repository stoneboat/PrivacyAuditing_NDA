#!/usr/bin/env python
"""
Gaussian Mechanism for Differential Privacy

This script computes the private average of a dataset using the Gaussian mechanism.
The data is first normalized to [0,1] range, then the private average is computed,
and finally the result is denormalized back to the original scale.

Example running commands:
python gaussian_mechanism.py \
  --data data/heart_failure_clinical_records_dataset.csv \
  --epsilon 1.0 \
  --delta 1e-5 \
  --samples 100
"""

import argparse
import numpy as np
import pandas as pd
import sys
from typing import Tuple, Optional, List

def compute_sigma(epsilon: float, delta: float) -> float:
    """
    Compute the noise scale (sigma) for the Gaussian mechanism.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        
    Returns:
        Noise scale (sigma)
    """
    # Using the analytic Gaussian mechanism formula
    # See: https://arxiv.org/abs/1805.06530
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon

def gaussian_mechanism(data: np.ndarray, epsilon: float, delta: float, num_samples: int = 1) -> List[np.ndarray]:
    """
    Apply Gaussian mechanism to compute private average.
    
    Args:
        data: Input data array (normalized to [0,1])
        epsilon: Privacy budget
        delta: Privacy parameter
        num_samples: Number of noisy copies to generate
        
    Returns:
        List of private averages
    """
    # Compute sensitivity (assuming data is normalized to [0,1])
    sensitivity = 1.0 / len(data)
    
    # Compute noise scale (sigma)
    sigma = compute_sigma(epsilon, delta)
    
    # Generate multiple noisy copies
    private_averages = []
    for _ in range(num_samples):
        noise = np.random.normal(0, sigma * sensitivity, size=data.shape[1])
        private_average = np.mean(data, axis=0) + noise
        private_averages.append(private_average)
    
    return private_averages

def load_and_validate_data(file_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load and validate the input data.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (original DataFrame, normalized numpy array)
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the data is empty or contains non-numeric columns
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if df.empty:
        raise ValueError("Input file is empty")
    
    # Check for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if not non_numeric_cols.empty:
        raise ValueError(f"Found non-numeric columns: {non_numeric_cols.tolist()}")
    
    return df

def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize data to [0,1] range.
    
    Args:
        data: Input data array
        
    Returns:
        Tuple of (normalized data, min values, max values)
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    # Handle case where max == min (constant column)
    mask = data_max != data_min
    normalized = np.zeros_like(data)
    normalized[:, mask] = (data[:, mask] - data_min[mask]) / (data_max[mask] - data_min[mask])
    normalized[:, ~mask] = 0.5  # Set constant columns to 0.5
    
    return normalized, data_min, data_max

def print_results(df: pd.DataFrame, original_avg: np.ndarray, 
                 private_averages: List[np.ndarray], epsilon: float, delta: float) -> None:
    """
    Print the results in a formatted table.
    
    Args:
        df: Original DataFrame (for column names)
        original_avg: Original average values
        private_averages: List of private average values
        epsilon: Privacy budget
        delta: Privacy parameter
    """
    # Convert list of private averages to numpy array for statistics
    private_array = np.array(private_averages)
    empirical_mean = np.mean(private_array, axis=0)
    empirical_std = np.std(private_array, axis=0)
    
    print("\nResults:")
    print("-" * 100)
    print(f"{'Column':<20} {'Original Average':<20} {'Empirical Mean':<20} {'Empirical Std':<20} {'Max Difference':<20}")
    print("-" * 100)
    
    for col, orig, mean, std in zip(df.columns, original_avg, empirical_mean, empirical_std):
        max_diff = np.max(np.abs(private_array[:, df.columns.get_loc(col)] - orig))
        print(f"{col:<20} {orig:<20.4f} {mean:<20.4f} {std:<20.4f} {max_diff:<20.4f}")
    
    print("-" * 100)
    print(f"Privacy budget (ε): {epsilon}")
    print(f"Privacy parameter (δ): {delta}")
    print(f"Number of samples: {len(private_averages)}")
    print(f"Maximum difference across all columns: {np.max(np.abs(private_array - original_avg)):.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Compute private average using Gaussian mechanism",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", required=True, help="Path to the dataset")
    parser.add_argument("--epsilon", type=float, default=1.0, 
                       help="Privacy budget (smaller values = more privacy)")
    parser.add_argument("--delta", type=float, default=1e-5,
                       help="Privacy parameter (probability of privacy failure)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of noisy copies to generate")
    
    args = parser.parse_args()

    try:
        # Load and validate data
        print("Loading data...")
        df = load_and_validate_data(args.data)
        X = df.to_numpy(dtype=np.float32)
        
        # Normalize data
        print("Normalizing data...")
        X_normalized, X_min, X_max = normalize_data(X)
        
        # Compute private averages
        print(f"Computing {args.samples} private averages...")
        private_averages = gaussian_mechanism(X_normalized, args.epsilon, args.delta, args.samples)
        
        # Denormalize the results
        private_averages_denormalized = [
            avg * (X_max - X_min) + X_min for avg in private_averages
        ]
        
        # Print results
        print_results(df, np.mean(X, axis=0), private_averages_denormalized, 
                     args.epsilon, args.delta)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 