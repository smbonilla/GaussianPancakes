# Author:   Sierra Bonilla
#! /bin/bash 

# make sure to have activated the virtual environment

pip show gaussian-norms &> /dev/null
if [ $? -eq 0 ]; then
    echo "Uninstalling existing gaussian-norms package..."
    pip uninstall gaussian-norms -y
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