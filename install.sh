#!/bin/bash

# Ensure pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 could not be found. Please install python3-pip."
    exit 1
fi

echo "Installing multi-agent coder..."

# Install the package in editable mode or normally
pip3 install -e .

if [ $? -eq 0 ]; then
    echo "Installation successful! You can now run 'agentchanti' from anywhere."
else
    echo "Installation failed."
    exit 1
fi
