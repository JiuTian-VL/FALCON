#!/bin/bash

# Step 0: Clean up
rm -rf dist

# Step 1: Change the package name to "jiutian-torch"
sed -i 's/name = "jiutian"/name = "jiutian-torch"/' pyproject.toml

# Step 2: Build the package
python -m build

# Step 3: Revert the changes in pyproject.toml to the original
sed -i 's/name = "jiutian-torch"/name = "jiutian"/' pyproject.toml

# Step 4: Upload to PyPI
python -m twine upload dist/*
