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
  --delta 1e-5
"""

import argparse
import numpy as np
import pandas as pd
import sys
from opacus.accountants import GaussianAccountant
from typing import Tuple, Optional

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

def gaussian_mechanism(data: np.ndarray, epsilon: float, delta: float) -> np.ndarray:
    """
    Apply Gaussian mechanism to compute private average.
    
    Args:
        data: Input data array (normalized to [0,1])
        epsilon: Privacy budget
        delta: Privacy parameter
        
    Returns:
        Private average of the data
    """
    # Compute sensitivity (assuming data is normalized to [0,1])
    sensitivity = 1.0 / len(data)
    
    # Compute noise scale (sigma)
    accountant = GaussianAccountant()
    sigma = accountant.compute_sigma(epsilon, delta)
    
    # Add Gaussian noise
    noise = np.random.normal(0, sigma * sensitivity, size=data.shape[1])
    private_average = np.mean(data, axis=0) + noise
    
    return private_average

def print_results(df: pd.DataFrame, original_avg: np.ndarray, 
                 private_avg: np.ndarray, epsilon: float, delta: float) -> None:
    """
    Print the results in a formatted table.
    
    Args:
        df: Original DataFrame (for column names)
        original_avg: Original average values
        private_avg: Private average values
        epsilon: Privacy budget
        delta: Privacy parameter
    """
    print("\nResults:")
    print("-" * 80)
    print(f"{'Column':<20} {'Original Average':<20} {'Private Average':<20} {'Difference':<20}")
    print("-" * 80)
    
    for col, orig, priv in zip(df.columns, original_avg, private_avg):
        diff = abs(orig - priv)
        print(f"{col:<20} {orig:<20.4f} {priv:<20.4f} {diff:<20.4f}")
    
    print("-" * 80)
    print(f"Privacy budget (ε): {epsilon}")
    print(f"Privacy parameter (δ): {delta}")
    print(f"Maximum difference: {np.max(np.abs(original_avg - private_avg)):.4f}")

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
    
    args = parser.parse_args()

    try:
        # Load and validate data
        print("Loading data...")
        df = load_and_validate_data(args.data)
        X = df.to_numpy(dtype=np.float32)
        
        # Normalize data
        print("Normalizing data...")
        X_normalized, X_min, X_max = normalize_data(X)
        
        # Compute private average
        print("Computing private average...")
        private_average = gaussian_mechanism(X_normalized, args.epsilon, args.delta)
        
        # Denormalize the result
        private_average_denormalized = private_average * (X_max - X_min) + X_min
        
        # Print results
        print_results(df, np.mean(X, axis=0), private_average_denormalized, 
                     args.epsilon, args.delta)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 