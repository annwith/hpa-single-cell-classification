#!/bin/bash

# Detect environment
if [[ $ENV == cenapad ]]; then
    echo "Environment: Cenapad"

    # unset CUDA_VISIBLE_DEVICES
    # export OMP_NUM_THREADS=8

    # MODULES=("python/3.8.11-gcc-9.4.0")

    # Activate virtual environment if it exists
    # echo "Activating virtual environment... ($HOME/dev/bin/activate)"
    # source $HOME/dev/bin/activate

else
    WORK_DIR=/home/juliana/Documentos/github-repositories/hpa-single-cell-classification
    
    MODULES=()  # No modules needed locally

    # Activate virtual environment if it exists
    echo "Activating virtual environment... ($WORK_DIR/.venv)"
    source $WORK_DIR/.venv/bin/activate
fi

# Load modules if necessary
if [[ ${#MODULES[@]} -gt 0 ]]; then
    for module in "${MODULES[@]}"; do
        module load $module
    done
fi

echo "Environment: $ENV"