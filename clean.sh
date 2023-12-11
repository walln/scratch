#!/bin/bash

# Remove the .venv directory
echo "Removing .venv directory..."
rm -rf .venv

# Remove the poetry.lock file
echo "Removing poetry.lock file..."
rm -f poetry.lock

# Find and remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# List all Poetry caches and clear them
echo "Listing and clearing all Poetry caches..."
poetry cache list | grep -v 'Configuration file exists at' | grep -v 'Consider moving TOML configuration files' | awk '{print $1}' | while read -r cache_name; do
    if [ ! -z "$cache_name" ]; then
        echo "Clearing cache: $cache_name"
        poetry cache clear --all $cache_name
    fi
done

echo "Cleanup completed."
