# Author:   Sierra Bonilla
#! /bin/bash 

# make sure to have activated the virtual environment

pip show diff-gaussian-rasterization &> /dev/null
if [ $? -eq 0 ]; then
    echo "Uninstalling existing depth-diff-gaussian-rasterization package..."
    pip uninstall diff-gaussian-rasterization -y
fi

# Navigate to the directory of this script
cd "$(dirname "$0")"

# Remove the build artifacts
echo "Cleaning up old build artifacts..."
rm -rf build/ dist/ *.egg-info

# Rebuild the package
echo "Rebuilding the package..."
pip install . 

echo "Done!"