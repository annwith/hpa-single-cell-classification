#PBS -N hpa-squeezenet
#PBS -q testegpu
#PBS -e hpa-squeezenet-errors.txt
#PBS -o hpa-squeezenet-logs.txt

#
# Train a model to perform multilabel classification over a WSSS dataset.
#

HOME=/home/lovelace/proj/proj1018/jmidlej

module load cudnn/8.2.0.53-11.3-gcc-9.3.0
module load python/3.8.11-gcc-9.4.0

ENV=cenapad
SCRATCH=$HOME
WORK_DIR=$HOME/hpa-single-cell-classification

PY=python     # path to python
PIP=pip       # path to PIP
DEVICES=0        # the GPUs used.
WORKERS=24       # number of workers spawn during dCRF refinement and evaluation.

source $HOME/dev/bin/activate
cd $WORK_DIR

# region Setup
# $PIP install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# $PIP install -r requirements.txt
# endregion

# Executar o train.py com os argumentos especificados
echo "=================================================================="
echo "[train started at $(date +'%Y-%m-%d %H:%M:%S')."
echo "=================================================================="

$PY train.py \
    --epochs 5 \
    --batch_size 64 \
    --weights_update 1 \
    --learning_rate 0.001 \
    --architecture resnet50 \
    --dataset_path /home/lovelace/proj/proj1018/jmidlej/datasets/kaggle_joined_resized_train \
    --labels_path /home/lovelace/proj/proj1018/jmidlej/datasets/train.csv \
    --pretrained_weights_path /home/lovelace/proj/proj1018/jmidlej/checkpoints/pretrained_resnet_checkpoint.pth \
    --save_checkpoint_path /home/lovelace/proj/proj1018/jmidlej/checkpoints/resnet_checkpoint.pth
