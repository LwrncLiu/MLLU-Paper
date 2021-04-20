#!/bin/bash
#SBATCH --job-name=layoutLM
#SBATCH --open-mode=append
#SBATCH --output=/scratch/fs1493/mlu_project/%j_%x.out
#SBATCH --error=/scratch/fs1493/mlu_project/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/env.sh
conda activate
python ./run_finetuning.py
"