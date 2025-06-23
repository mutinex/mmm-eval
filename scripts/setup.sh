#!/bin/bash

# Setup script for mmm-eval development environment
# This script sets up asdf, installs Python, and configures Poetry
#
# Note: If you get a "permission denied" error, run:
#   chmod +x scripts/setup.sh
# This is needed because git doesn't preserve file permissions in shared repositories.

set -e  # Exit on any error

echo "🚀 Setting up mmm-eval development environment..."

# Check if asdf is installed
if ! command -v asdf &> /dev/null; then
    echo "❌ asdf is not installed. Please install it first:"
    echo "   brew install asdf"
    echo "   Then add it to your shell config and restart your terminal."
    exit 1
fi

# Check if Python plugin is installed
if ! asdf plugin list | grep -q python; then
    echo "📦 Adding Python plugin to asdf..."
    asdf plugin add python
fi

# Install Python version from .tool-versions
echo "🐍 Installing Python version from .tool-versions..."
asdf install

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "📦 Adding Poetry shell plugin..."
    poetry self add poetry-plugin-shell
    echo "✅ Poetry installed! You may need to restart your terminal or run 'source ~/.zshrc'"
fi

# Configure Poetry to use asdf Python
echo "🔧 Configuring Poetry to use asdf Python..."
poetry env use $(asdf which python)

# Install dependencies
echo "📦 Installing project dependencies..."
poetry install

echo "✅ Setup complete!"
echo ""
echo "To activate the Poetry environment, run:"
echo "  poetry shell"
echo ""
echo "To run tests:"
echo "  poetry run pytest"
echo ""
echo "To run the CLI:"
echo "  poetry run mmm-eval --help" 