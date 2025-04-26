# Privacy Auditing Project

This project implements and analyzes privacy mechanisms, focusing on the Gaussian mechanism for differential privacy and neural network privacy testing.

## Project Structure

```
.
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── scripts/
│   ├── run_gaussian.sh
│   ├── run_gaussian_experiments.sh
│   └── test_nn.py
├── tmp_data/
│   ├── analysis_results/
│   │   └── distribution_comparison_epsilon_*.png
│   ├── private_averages_epsilon_*.csv
│   └── debug_private_averages_epsilon_*.png
├── gaussian_mechanism.py
├── analyze_distributions.py
└── README.md
```

## Requirements

- Python 3.x
- Required Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - torch

## Gaussian Mechanism Analysis

### Running the Gaussian Mechanism

1. Basic usage with default parameters:
```bash
bash scripts/run_gaussian.sh
```

2. Custom parameters:
```bash
bash scripts/run_gaussian.sh --epsilon 1.0 --delta 1e-5 --samples 1000
```

3. Run experiments with different epsilon values:
```bash
bash scripts/run_gaussian_experiments.sh
```

### Output Files

1. Private Averages:
   - Location: `tmp_data/private_averages_epsilon_*.csv`
   - Format: CSV with columns:
     - `age`: The age value (original or noised)
     - `id`: Dataset identifier (0 for original dataset, 1 for neighboring dataset)

2. Debug Plots:
   - Location: `tmp_data/debug_private_averages_epsilon_*.png`
   - Shows empirical vs theoretical distributions for both datasets

3. Analysis Results:
   - Location: `tmp_data/analysis_results/distribution_comparison_epsilon_*.png`
   - Compares distributions of original and neighboring datasets

### Analyzing Results

To analyze the distributions:
```bash
python analyze_distributions.py --input tmp_data/private_averages_epsilon_1.0.csv
```

## Neural Network Privacy Testing

### Running the Test

```bash
python scripts/test_nn.py
```

This script:
1. Loads a pre-trained model
2. Tests privacy properties
3. Generates analysis results

### Output Files

- Model checkpoints and test results are saved in the `tmp_data` directory

## Notes

- The `tmp_data` directory is automatically created if it doesn't exist
- All output files are saved in appropriate subdirectories under `tmp_data`
- The analysis results include both visualizations and statistical comparisons

