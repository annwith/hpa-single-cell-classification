#PBS -N hpa-squeezenet
#PBS -q testegpu
#PBS -e hpa-squeezenet-errors.txt
#PBS -o hpa-squeezenet-logs.txt

#
# Train a model to perform multilabel classification over a WSSS dataset.
#

# Load ENV variables
runners/config/env.sh

WORK_DIR=/home/juliana/Documentos/github-repositories/hpa-single-cell-classification

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# $PIP install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# $PIP install -r requirements.txt

# Training parameters
BATCH_SIZE=32

# Dataset parameters
DATASET_NAME="publichpa"
DATASET_CHANNELS=1
#DATASET_PATH="/mnt/ssd/hpa-single-cell-image-classification/join_resized_train"
#LABELS_PATH="/mnt/ssd/hpa-single-cell-image-classification/train.csv"
DATASET_PATH="/mnt/ssd/hpa-single-cell/train"
LABELS_PATH="/mnt/ssd/hpa-single-cell/train.csv"
PUBLICHPA_LABELS_PATH="/mnt/ssd/hpa-single-cell/publichpa.csv"

# Train the model
compute () {
    echo "\n=================================================================="
    echo "[train started at $(date +'%Y-%m-%d %H:%M:%S')]."
    echo "==================================================================\n"

    $PY $WORK_DIR/compute_mean_std.py \
    --dataset_channels $DATASET_CHANNELS \
    --dataset_path $DATASET_PATH \
    --labels_path $LABELS_PATH \
    --publichpa_labels_path $PUBLICHPA_LABELS_PATH \
    --batch_size $BATCH_SIZE
}

compute