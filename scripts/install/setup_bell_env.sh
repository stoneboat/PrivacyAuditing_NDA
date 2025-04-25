#!/bin/bash

# Get the project root directory (two levels up from this script)
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")
PARENT_DIR=$(dirname "$PROJECT_ROOT")

# Source the Anaconda environment
source /apps/spack/bell/apps/anaconda/2020.11-py38-gcc-4.8.5-nhzhrm2/etc/profile.d/conda.sh

# Create virtual environment in parent directory using the Anaconda Python
/apps/spack/bell/apps/anaconda/2020.11-py38-gcc-4.8.5-nhzhrm2/bin/python -m venv "$PARENT_DIR/PrivacyAuditing_venv" --copies

# Activate virtual environment
source "$PARENT_DIR/PrivacyAuditing_venv/bin/activate"

# Verify Python path and version
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# List installed packages
echo "Installed packages:"
pip list

echo "Virtual environment setup complete!"
echo "Virtual environment location: $PARENT_DIR/PrivacyAuditing_venv"
echo "To activate the environment, run: source $PARENT_DIR/PrivacyAuditing_venv/bin/activate" 