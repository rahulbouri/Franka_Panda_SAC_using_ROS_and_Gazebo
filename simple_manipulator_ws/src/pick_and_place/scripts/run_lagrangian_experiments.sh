#!/bin/bash

# Lagrangian Experiments Script for Franka Panda
# Runs the experiments mentioned in the README

set -e  # Exit on any error

echo "Starting Lagrangian Experiments for Franka Panda..."

# Create necessary directories
mkdir -p logs/lagrangian
mkdir -p models

# Change to the scripts directory
cd "$(dirname "$0")"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Experiment 1: Joint Identification Test
echo "=== Running Joint Identification Experiment ==="
python3 deep_lagrangian_experiments.py --mode id_test --episodes 200

# Experiment 2: Structure Analysis
echo "=== Running Lagrangian Structure Analysis ==="
python3 deep_lagrangian_experiments.py --mode structure_analysis --episodes 50

# Experiment 3: Residual Controller Training
echo "=== Training Residual Controller ==="
python3 residual_controller.py --train --episodes 1000

# Experiment 4: Full Lagrangian Experiment
echo "=== Running Full Lagrangian Experiment ==="
python3 deep_lagrangian_experiments.py --mode full_experiment --episodes 500

echo "All Lagrangian experiments completed!"
echo "Results saved in:"
echo "  - logs/lagrangian/"
echo "  - models/"

# Display results summary
echo "=== Experiment Results Summary ==="
if [ -f "logs/lagrangian/delan_model.pth" ]; then
    echo "✓ DeLaN model trained successfully"
else
    echo "✗ DeLaN model training failed"
fi

if [ -f "models/residual_controller.pth" ]; then
    echo "✓ Residual controller trained successfully"
else
    echo "✗ Residual controller training failed"
fi

echo "Lagrangian experiments completed!"
