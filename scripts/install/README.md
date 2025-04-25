# Installation Scripts

This directory contains scripts for setting up the development environment on Bell cluster.

## Files

- `setup_bell_env.sh`: Creates a virtual environment using Bell's Anaconda Python installation

## Prerequisites

- Access to Bell cluster
- Anaconda environment at `/apps/spack/bell/apps/anaconda/2020.11-py38-gcc-4.8.5-nhzhrm2/`

## Usage

1. Make the script executable:
   ```bash
   chmod +x setup_bell_env.sh
   ```

2. Run the script:
   ```bash
   ./setup_bell_env.sh
   ```

## What the Script Does

1. Creates a virtual environment in the parent directory of the project root
2. Uses Bell's Anaconda Python installation (Python 3.8)
3. Copies all existing packages from the Anaconda environment
4. Verifies the Python installation and lists installed packages

## Environment Location

The virtual environment will be created at:
```
../../PrivacyAuditing_venv
```
(relative to this script's location)

## Activating the Environment

After setup, activate the environment with:
```bash
source ../../PrivacyAuditing_venv/bin/activate
```

## Notes

- The script uses the `--copies` flag to ensure all packages are copied from the Anaconda environment
- The virtual environment is created outside the project directory to keep the project clean
- All existing packages (numpy, pandas, torch, etc.) will be available in the virtual environment
- The script automatically determines the correct parent directory regardless of where it's run from 