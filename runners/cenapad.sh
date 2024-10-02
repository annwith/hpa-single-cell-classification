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
$PY train.py \
    --epochs 10 \
    --batch_size 8 \
    --weights_update 8 \
    --lr 0.001 \
    --model squeezenet-cam \
    --dataset_dir /mnt/ssd/hpa-single-cell-image-classification/join_resized_train \
    --labels_csv /mnt/ssd/hpa-single-cell-image-classification/train.csv \
    --checkpoint /mnt/ssd/checkpoints/squeezenet_checkpoint.pth
