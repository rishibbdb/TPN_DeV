#!/bin/bash --login
#SBATCH --ntasks=1       # number of CPUs
#SBATCH --gpus=h200:1   # number of GPUs
# SBATCH --mem-per-cpu=20G # memory for CPUs
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name hybrid_example

source /mnt/home/baburish/miniconda3/miniconda/etc/profile.d/conda.sh
conda activate 3pandelnet
module load TensorFlow/2.13.0-foss-2023a
python /mnt/home/baburish/jax/TriplePandelReco_JAX/convert_npy_to_tfrecords.py | tee /mnt/home/baburish/jax/TriplePandelReco_JAX/npyconversion-log.txt
