#!/bin/bash
#SBATCH --nodes=1           #Numero de Nós
#SBATCH --ntasks-per-node=1 #Numero de tarefas por Nó
#SBATCH --ntasks=1          #Numero de tarefas
#SBATCH -p gdl              #Fila (partition) a ser utilizada
#SBATCH -J GDL-teste-script #Nome job
#SBATCH --exclusive         #Utilização exclusiva dos nós durante a execução do job

#Exibe os nos alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

cd $SLURM_SUBMIT_DIR

#Configura o módulo de Deep Learning
module load deepl/deeplearn-py3.7

#acessa o diretório onde o script está localizado 
cd /prj/lerdl/zanoni.dias/juliana

#executa o script
python python3 train.py --dataset_dir /prj/lerdl/zanoni.dias/juliana/kaggle_resized_train_dataset --labels_csv /prj/lerdl/zanoni.dias/juliana/train.csv
