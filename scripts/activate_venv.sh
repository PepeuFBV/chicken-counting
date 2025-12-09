#!/usr/bin/env bash

# Quick helper to activate the project's `.venv`.
# Usage (recommended):
#   source scripts/activate_venv.sh
# If you execute this script instead of sourcing it, it will print instructions
# because activation must happen in the current shell.

VENV_DIR=".venv"

# detect whether the script was sourced (works for bash/zsh)
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    SOURCED=1
else
    SOURCED=0
fi

if [ "$SOURCED" -ne 1 ]; then
    echo "This script must be sourced to activate the venv in your current shell."
    echo "Run: source scripts/activate_venv.sh"
    echo "Or activate manually: source $VENV_DIR/bin/activate"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at '$VENV_DIR'."
    echo "Create it with: python3 -m venv $VENV_DIR"
    return 1
fi

ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    echo "Activate script not found: $ACTIVATE_SCRIPT"
    return 1
fi

# source the venv activation script
source "$ACTIVATE_SCRIPT"

echo "Activated virtualenv: $VENV_DIR"
