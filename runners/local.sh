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
EPOCHS=1
BATCH_SIZE=4
ACCUMULATE_STEPS=8
LEARNING_RATE=0.01

# Model parameters
ARCHITECTURE="resnet50"
PRETRAINED_WEIGHTS_PATH=none

# Dataset parameters
DATASET_NAME="kaggle_joined_resized"
DATASET_CHANNELS=4
DATASET_PATH="/mnt/ssd/hpa-single-cell-image-classification/join_resized_train"
LABELS_PATH="/mnt/ssd/hpa-single-cell-image-classification/train.csv"

CLASS_WEIGHTS=0.1,1.0,0.5,1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,10.0,1.0,0.5,0.5,5.0,0.2,0.5,1.0

# Checkpoint parameters
RESUME_CHECKPOINT_PATH=none
SAVE_CHECKPOINT_PATH="/mnt/ssd/checkpoints/resnet_checkpoint.pth"

# WandB parameters
WANDB_PROJECT="hpa-single-cell-classification"
WANDB_ENTITY="lerdl"
WANDB_RUN_NAME=$DATASET_NAME-$ARCHITECTURE-$BATCH_SIZE-$ACCUMULATE_STEPS-$LEARNING_RATE-$(date +'%Y.%m.%d_%H:%M:%S')
WANDB_MODE="offline"

echo "WandB run name: $WANDB_RUN_NAME"

# Train the model
train_model () {
    echo "\n=================================================================="
    echo "[train started at $(date +'%Y-%m-%d %H:%M:%S')]."
    echo "==================================================================\n"

    $PY $WORK_DIR/main.py \
    --dataset_channels $DATASET_CHANNELS \
    --dataset_path $DATASET_PATH \
    --labels_path $LABELS_PATH \
    --class_weights $CLASS_WEIGHTS \
    --architecture $ARCHITECTURE \
    --pretrained_weights_path $PRETRAINED_WEIGHTS_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --learning_rate $LEARNING_RATE \
    --save_checkpoint_path $SAVE_CHECKPOINT_PATH \
    --resume_checkpoint_path $RESUME_CHECKPOINT_PATH \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --wandb_run_name $WANDB_RUN_NAME \
    --wandb_mode $WANDB_MODE
}

train_model