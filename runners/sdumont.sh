#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_gpu_dev
#SBATCH -J hpa-test
#SBATCH -o /scratch/lerdl/zanoni.dias/logs/hpa-single-cell-classification/%j-hpa-project.out
#SBATCH --time=00:20:00

#
# Train a model to perform multilabel classification.
#

# export OMP_NUM_THREADS=4
nodeset -e $SLURM_JOB_NODELIST
# module load sequana/current
module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana

WORK_DIR=$SCRATCH/hpa-single-cell-classification

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

# Set up the environment
PY=python3     # path to python
PIP=pip       # path to PIP

# $PIP install -r requirements.txt

# Training parameters
EPOCHS=5
BATCH_SIZE=16
ACCUMULATE_STEPS=2
LEARNING_RATE=0.01
OPTIMIZER_NAME="sgd"

# Model parameters
ARCHITECTURE="resnet50"
PRETRAINED_WEIGHTS_PATH=/scratch/lerdl/zanoni.dias/hpa-single-cell-classification/weights/resnet50_imagenet_weights.pth

# Dataset parameters
# DATASET_NAME="kaggle"
# DATASET_CHANNELS=4
# DATASET_PATH="/home/lovelace/proj/proj1018/jmidlej/datasets/kaggle_joined_resized_train"
# LABELS_PATH="/home/lovelace/proj/proj1018/jmidlej/datasets/train.csv"
# PUBLICHPA_LABELS_PATH=none
# IMAGE_NORMALIZATION="imagenet"

DATASET_NAME="publichpa"
DATASET_CHANNELS=1
DATASET_PATH="/scratch/lerdl/zanoni.dias/datasets/hpa-single-cell/train"
LABELS_PATH="/scratch/lerdl/zanoni.dias/datasets/hpa-single-cell/train.csv"
PUBLICHPA_LABELS_PATH="/scratch/lerdl/zanoni.dias/datasets/hpa-single-cell/publichpa.csv"
IMAGE_NORMALIZATION="imagenet"

CLASS_WEIGHTS=0.1,1.0,0.5,1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,10.0,1.0,0.5,0.5,5.0,0.2,0.5,1.0

# Checkpoint parameters
RESUME_CHECKPOINT_PATH=none
SAVE_CHECKPOINT_PATH="/scratch/lerdl/zanoni.dias/checkpoints"

# WandB parameters
EID=1
WANDB_PROJECT="hpa-single-cell-classification"
WANDB_ENTITY="lerdl"
WANDB_RUN_NAME=$DATASET_NAME-$ARCHITECTURE-b$BATCH_SIZE-acc$ACCUMULATE_STEPS-lr$LEARNING_RATE-$OPTIMIZER_NAME-eid$EID-$(date +'%Y%m%d')
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
    --publichpa_labels_path $PUBLICHPA_LABELS_PATH \
    --image_normalization $IMAGE_NORMALIZATION \
    --class_weights $CLASS_WEIGHTS \
    --architecture $ARCHITECTURE \
    --pretrained_weights_path $PRETRAINED_WEIGHTS_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulate_steps $ACCUMULATE_STEPS \
    --learning_rate $LEARNING_RATE \
    --optimizer_name $OPTIMIZER_NAME \
    --save_checkpoint_path $SAVE_CHECKPOINT_PATH \
    --resume_checkpoint_path $RESUME_CHECKPOINT_PATH \
    --wandb_project $WANDB_PROJECT \
    --wandb_entity $WANDB_ENTITY \
    --wandb_run_name $WANDB_RUN_NAME \
    --wandb_mode $WANDB_MODE
}

train_model