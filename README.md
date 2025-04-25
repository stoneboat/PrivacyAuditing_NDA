## Project Structure

```
.
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── scripts/
│   └── run_gaussian.sh
└── gaussian_mechanism.py
```

## Running the Script

The project includes a shell script to easily run the Gaussian mechanism with different privacy parameters.

### Using the Run Script

1. Make the script executable:
```bash
chmod +x scripts/run_gaussian.sh
```

2. Run with default parameters:
```bash
./scripts/run_gaussian.sh
```
This will run with:
- ε (epsilon) = 1.0
- δ (delta) = 1e-5
- Number of samples = 1000

3. Run with custom parameters:
```bash
./scripts/run_gaussian.sh --epsilon 0.5 --delta 1e-6 --samples 2000
```

### Parameters Explanation

- `--epsilon`: Privacy budget (ε)
  - Lower values provide stronger privacy guarantees
  - Default: 1.0
  - Range: Typically between 0.1 and 10

- `--delta`: Failure probability (δ)
  - Probability that the privacy guarantee fails
  - Default: 1e-5
  - Range: Typically between 1e-10 and 1e-3

- `--samples`: Number of samples
  - Number of times to run the mechanism
  - Default: 1000
  - Higher values provide more accurate results

## Output

The script will output:
- Original data statistics
- Noisy statistics after applying the Gaussian mechanism
- Privacy parameters used
- Standard deviation of the added noise

## Example Output

```
Original Average: 0.1234
Noisy Average: 0.1256
Standard Deviation: 0.0023
Privacy Parameters:
- Epsilon: 1.0
- Delta: 1e-5
```

