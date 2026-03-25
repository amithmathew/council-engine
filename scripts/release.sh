#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Get version from pyproject.toml
VERSION=$(python -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    print(tomllib.load(f)['project']['version'])
")

echo "==> Releasing council-engine v${VERSION}"

# Clean previous builds
echo "==> Cleaning dist/"
rm -rf dist/ build/ *.egg-info

# Build
echo "==> Building..."
python -m build

# Verify
echo "==> Checking with twine..."
twine check dist/*

# Upload
echo "==> Uploading to PyPI..."
twine upload dist/*

echo "==> Done! council-engine v${VERSION} published to PyPI"
echo "    https://pypi.org/project/council-engine/${VERSION}/"
