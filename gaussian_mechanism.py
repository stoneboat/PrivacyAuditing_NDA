#!/usr/bin/env python
"""
Gaussian Mechanism for Differential Privacy

This script computes the private average of a dataset using the Gaussian mechanism.
The noise is added directly based on the sensitivity of the mean query.

Example running commands:
python gaussian_mechanism.py \
  --data data/heart_failure_clinical_records_dataset.csv \
  --epsilon 1.0 \
  --delta 1e-5 \
  --samples 100 \
  --output tmp_data/private_averages.csv

Output Parser Helper:
The output CSV file contains the following columns:
- age: The age value (original or noised)
- DEATH_EVENT: The death event value (original or noised)
- id: Dataset identifier
  - 0: Record from original dataset (X)
  - 1: Record from neighboring dataset (X')

The script also prints:
1. Dataset information:
   - Original dataset size
   - Neighboring dataset size (after removing 5 oldest records)
2. Computation times for both datasets
3. Statistical results for both datasets:
   - Original averages
   - Empirical means
   - Standard deviations
   - Maximum differences
4. Privacy parameters used:
   - Epsilon (ε)
   - Delta (δ)
"""

import argparse
import numpy as np
import pandas as pd
import sys
from typing import Tuple, Optional, List
import time
import matplotlib.pyplot as plt
import os

def compute_sigma(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Compute the noise scale (sigma) for the Gaussian mechanism.
    
    Args:
        epsilon: Privacy budget
        delta: Privacy parameter
        sensitivity: L2 sensitivity of the query
        
    Returns:
        Noise scale (sigma)
    """
    # Using the analytic Gaussian mechanism formula
    # See: https://arxiv.org/abs/1805.06530
    return np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon

def compute_sensitivity(X: np.ndarray, X_prime: np.ndarray) -> float:
    """
    Compute the L2 sensitivity between two datasets.
    
    Args:
        X: Original dataset
        X_prime: Neighboring dataset
        
    Returns:
        L2 sensitivity between the datasets
    """
    # Compute the mean of each dataset
    mean_X = np.mean(X, axis=0)
    mean_X_prime = np.mean(X_prime, axis=0)
    
    # Compute L2 distance between the means
    sensitivity = np.linalg.norm(mean_X - mean_X_prime, ord=2)
    return sensitivity

def plot_empirical_distribution(private_averages: np.ndarray, true_mean: np.ndarray, 
                              sigma: float, epsilon: float, delta: float,
                              output_path: str, title_suffix: str = "") -> None:
    """
    Plot both empirical and theoretical Gaussian distributions of private averages.
    
    Args:
        private_averages: Array of private averages
        true_mean: True mean of the data
        sigma: Noise scale
        epsilon: Privacy budget
        delta: Privacy parameter
        output_path: Path to save the plot
        title_suffix: Additional text to add to the title
    """
    # Generate x values for the theoretical plot
    x = np.linspace(true_mean[0] - 4*sigma, true_mean[0] + 4*sigma, 1000)
    
    # Compute the theoretical Gaussian PDF
    pdf = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - true_mean[0])/sigma)**2)
    
    plt.figure(figsize=(10, 6))
    
    # Plot empirical distribution
    plt.hist(private_averages[:, 0], bins=30, density=True, alpha=0.5, 
            label='Empirical Distribution')
    
    # Plot theoretical distribution
    plt.plot(x, pdf, 'r-', linewidth=2, label='Theoretical Gaussian')
    
    # Add reference lines
    plt.axvline(true_mean[0], color='k', linestyle='--', label=f'True Mean: {true_mean[0]:.2f}')
    plt.axvline(true_mean[0] + sigma, color='g', linestyle=':', label=f'Mean ± σ: {sigma:.2f}')
    plt.axvline(true_mean[0] - sigma, color='g', linestyle=':')
    
    plt.title(f'Empirical vs Theoretical Distribution {title_suffix}(ε={epsilon}, δ={delta})')
    plt.xlabel('Age')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved distribution comparison plot to: {output_path}")

def gaussian_mechanism(data: np.ndarray, epsilon: float, delta: float, 
                      sensitivity: float, num_samples: int = 1) -> List[np.ndarray]:
    """
    Apply Gaussian mechanism to compute private average.
    
    Args:
        data: Input data array
        epsilon: Privacy budget
        delta: Privacy parameter
        sensitivity: L2 sensitivity between datasets
        num_samples: Number of noisy copies to generate
        
    Returns:
        List of private averages
    """
    # Compute noise scale (sigma)
    sigma = compute_sigma(epsilon, delta, sensitivity)
    
    # Compute mean once
    true_mean = np.mean(data, axis=0)
    
    # Generate all noise samples at once using numpy's random.normal
    # Shape: (num_samples, data.shape[1])
    noise = np.random.normal(0, sigma, size=(num_samples, data.shape[1]))
    
    # Add noise to mean using broadcasting
    private_averages = true_mean + noise
    
    return private_averages.tolist()

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

def save_private_averages_to_csv(df: pd.DataFrame, private_averages: List[np.ndarray], 
                               private_averages_prime: List[np.ndarray],
                               output_path: str = "tmp_data/private_averages.csv") -> None:
    """
    Save private averages from both original and neighboring datasets to a single CSV file.
    
    Args:
        df: Original DataFrame (for column names)
        private_averages: List of private average values from original dataset
        private_averages_prime: List of private average values from neighboring dataset
        output_path: Path to save the CSV file
    """
    # Convert lists of private averages to DataFrames
    private_df = pd.DataFrame(private_averages, columns=df.columns)
    private_df_prime = pd.DataFrame(private_averages_prime, columns=df.columns)
    
    # Add id column to distinguish between datasets
    private_df['id'] = 0  # 0 for original dataset
    private_df_prime['id'] = 1  # 1 for neighboring dataset
    
    # Combine both DataFrames
    combined_df = pd.concat([private_df, private_df_prime], ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\nPrivate averages from both datasets saved to: {output_path}")

def noised_average_experiment(data_path: str, epsilon: float = 1.0, delta: float = 1e-5, 
                            samples: int = 100, output_path: str = "tmp_data/private_averages.csv") -> None:
    """
    Run a complete experiment of computing noised averages using the Gaussian mechanism.
    
    Args:
        data_path: Path to the input dataset
        epsilon: Privacy budget (smaller values = more privacy)
        delta: Privacy parameter (probability of privacy failure)
        samples: Number of noisy copies to generate
        output_path: Path to save the private averages CSV file
    """
    try:
        # Create tmp_data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load and validate data
        print("Loading data...")
        df = load_and_validate_data(data_path)
        
        # Select only age column
        selected_columns = ['age']
        df = df[selected_columns]
        
        # Create neighboring dataset X' by removing 5 oldest records
        m = 5
        df_prime = df.copy()
        oldest_indices = df_prime['age'].nlargest(m).index
        df_prime = df_prime.drop(oldest_indices)
        print(f"\nCreated neighboring dataset by removing {m} oldest records")
        print(f"Original dataset size: {len(df)}")
        print(f"Neighboring dataset size: {len(df_prime)}")
        
        # Convert both datasets to numpy arrays
        X = df.to_numpy(dtype=np.float32)
        X_prime = df_prime.to_numpy(dtype=np.float32)
        
        # Compute sensitivity between X and X'
        sensitivity = compute_sensitivity(X, X_prime)
        print(f"\nComputed L2 sensitivity between datasets: {sensitivity:.4f}")
        
        # Compute private averages for both datasets
        print(f"\nComputing {samples} private averages for original dataset...")
        start_time = time.time()
        private_averages = gaussian_mechanism(X, epsilon, delta, sensitivity, samples)
        end_time = time.time()
        print(f"Computation time: {end_time - start_time:.2f} seconds")
        
        print(f"\nComputing {samples} private averages for neighboring dataset...")
        start_time = time.time()
        private_averages_prime = gaussian_mechanism(X_prime, epsilon, delta, sensitivity, samples)
        end_time = time.time()
        print(f"Computation time: {end_time - start_time:.2f} seconds")
        
        # Convert to numpy arrays for plotting
        private_averages = np.array(private_averages)
        private_averages_prime = np.array(private_averages_prime)
        
        # Plot empirical distributions
        # plot_empirical_distribution(
        #     private_averages, np.mean(X, axis=0), 
        #     compute_sigma(epsilon, delta, sensitivity),
        #     epsilon, delta,
        #     f'tmp_data/debug_private_averages_epsilon_{epsilon}.png',
        #     "(Original Dataset) "
        # )
        
        # plot_empirical_distribution(
        #     private_averages_prime, np.mean(X_prime, axis=0),
        #     compute_sigma(epsilon, delta, sensitivity),
        #     epsilon, delta,
        #     f'tmp_data/debug_private_averages_prime_epsilon_{epsilon}.png',
        #     "(Neighboring Dataset) "
        # )
        
        # Print results for both datasets
        print("\nResults for original dataset:")
        print_results(df, np.mean(X, axis=0), private_averages.tolist(), epsilon, delta)
        
        print("\nResults for neighboring dataset:")
        print_results(df_prime, np.mean(X_prime, axis=0), private_averages_prime.tolist(),
                     epsilon, delta)
        
        # Save private averages to CSV
        save_private_averages_to_csv(df, private_averages.tolist(), private_averages_prime.tolist(), output_path)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

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
    parser.add_argument("--output", type=str, default="tmp_data/private_averages.csv",
                       help="Path to save the private averages CSV file")
    
    args = parser.parse_args()
    
    # Run the experiment
    noised_average_experiment(
        data_path=args.data,
        epsilon=args.epsilon,
        delta=args.delta,
        samples=args.samples,
        output_path=args.output
    )

if __name__ == "__main__":
    main() 