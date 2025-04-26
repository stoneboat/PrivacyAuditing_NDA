#!/usr/bin/env python
"""
Analyze distributions of private averages from Gaussian mechanism experiments.

This script reads the output CSV files from the Gaussian mechanism experiments
and analyzes the empirical distributions of the age values for both original
and neighboring datasets.

Example running commands:
python analyze_distributions.py --input tmp_data/private_averages_epsilon_1.0.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Tuple
import seaborn as sns

def load_data(input_file: str) -> pd.DataFrame:
    """
    Load the private averages data from a CSV file.
    
    Args:
        input_file: Path to the input CSV file
        
    Returns:
        DataFrame containing the private averages
    """
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded data from: {input_file}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")

def compute_empirical_distribution(data: pd.DataFrame, column: str, id_value: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical distribution of a column for a specific dataset.
    
    Args:
        data: DataFrame containing the private averages
        column: Column to analyze
        id_value: Dataset identifier (0 for original, 1 for neighboring)
        
    Returns:
        Tuple of (bin_centers, densities) for the empirical distribution
    """
    # Filter data for the specified dataset
    dataset_data = data[data['id'] == id_value][column]
    
    # Compute histogram with density=True
    counts, bin_edges = np.histogram(dataset_data, bins=30, density=True)
    
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, counts

def plot_distribution(data: pd.DataFrame, column: str, epsilon: float, output_path: str) -> None:
    """
    Plot the empirical distributions of both X and X' together using KDE curves.
    
    Args:
        data: DataFrame containing both datasets
        column: Column to analyze
        epsilon: Privacy budget
        output_path: Path to save the plot
    """
    # Get data for both datasets
    data_0 = data[data['id'] == 0][column]
    data_1 = data[data['id'] == 1][column]
    
    plt.figure(figsize=(12, 6))
    
    # Plot KDE curves
    sns.kdeplot(data=data_0, color='blue', label='Original Dataset (X)', linewidth=2)
    sns.kdeplot(data=data_1, color='red', label='Neighboring Dataset (X\')', linewidth=2)
    
    # Add mean lines
    mean_0 = data_0.mean()
    mean_1 = data_1.mean()
    plt.axvline(mean_0, color='blue', linestyle='--', alpha=0.5, label=f'Mean X: {mean_0:.2f}')
    plt.axvline(mean_1, color='red', linestyle='--', alpha=0.5, label=f'Mean X\': {mean_1:.2f}')
    
    # Compute and display difference statistics
    diff_mean = abs(mean_0 - mean_1)
    
    plt.title(f'Distribution Comparison (ε={epsilon})\n'
             f'Mean Difference: {diff_mean:.2f}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")
    
    # Print detailed statistics
    print("\nDistribution Comparison Statistics:")
    print("-" * 50)
    print(f"Original Dataset (X):")
    print(f"  Mean: {mean_0:.2f}")
    print(f"  Std: {data_0.std():.2f}")
    print(f"Neighboring Dataset (X'):")
    print(f"  Mean: {mean_1:.2f}")
    print(f"  Std: {data_1.std():.2f}")
    print(f"\nDifferences:")
    print(f"  Mean Difference: {diff_mean:.2f}")

def analyze_file(input_file: str, output_dir: str) -> None:
    """
    Analyze the distributions in a single input file.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the analysis results
    """
    try:
        # Create tmp_data directory if it doesn't exist
        os.makedirs('tmp_data', exist_ok=True)
        
        # Create analysis_results directory under tmp_data
        analysis_dir = os.path.join('tmp_data', 'analysis_results')
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Load data
        df = load_data(input_file)
        
        # Extract epsilon from filename
        epsilon = float(input_file.split('epsilon_')[1].split('.csv')[0])
        
        # Plot distributions
        output_path = os.path.join(analysis_dir, f'distribution_comparison_epsilon_{epsilon}.png')
        plot_distribution(df, 'age', epsilon, output_path)
        
    except Exception as e:
        print(f"Error analyzing file {input_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze distributions of private averages from Gaussian mechanism experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="tmp_data/analysis_results",
                       help="Directory to save the analysis results")
    
    args = parser.parse_args()
    
    # Analyze the input file
    analyze_file(args.input, args.output_dir)

if __name__ == "__main__":
    main() 