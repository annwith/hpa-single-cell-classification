#PBS -N hpa-test
#PBS -q testegpu
#PBS -e logs/test.err
#PBS -o logs/test.log

#
# Train a model to perform multilabel classification.
#

ENV=cenapad
SCRATCH=$HOME
WORK_DIR=$HOME/hpa-single-cell-classification

unset CUDA_VISIBLE_DEVICES
# export OMP_NUM_THREADS=8

module load python/3.8.11-gcc-9.4.0

# Activate virtual environment if it exists
echo "Activating virtual environment... ($HOME/dev/bin/activate)"
source $HOME/dev/bin/activate

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# $PIP install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# $PIP install -r requirements.txt

# Training parameters
EPOCHS=5
BATCH_SIZE=16
ACCUMULATE_STEPS=4
LEARNING_RATE=0.01

# Model parameters
ARCHITECTURE="resnet50"
PRETRAINED_WEIGHTS_PATH=/home/lovelace/proj/proj1018/jmidlej/hpa-single-cell-classification/weights/resnet50_imagenet_weights.pth

# Dataset parameters
DATASET_NAME="kaggle_joined_resized"
DATASET_CHANNELS=4
DATASET_PATH="/home/lovelace/proj/proj1018/jmidlej/datasets/kaggle_joined_resized_train"
LABELS_PATH="/home/lovelace/proj/proj1018/jmidlej/datasets/train.csv"

CLASS_WEIGHTS=0.1,1.0,0.5,1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,10.0,1.0,0.5,0.5,5.0,0.2,0.5,1.0

# Checkpoint parameters
RESUME_CHECKPOINT_PATH=none
SAVE_CHECKPOINT_PATH="/home/lovelace/proj/proj1018/jmidlej/checkpoints

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