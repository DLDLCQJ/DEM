#!/bin/bash
#SBATCH --job-name=DEM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 
#SBATCH --gres=gpu:8 
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -o %j.out
#SBATCH -e %j.err

# --- Load Environment ---
module load anaconda/3
conda activate /xxx/.conda/envs/test_py38

# --- DDP Setup ---
export WORLD_SIZE=$SLURM_NTASKS 
export RANK=$SLURM_PROCID 
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$(shuf -i 10000-65535 -n1 | head -n1)  # Random port

# Verify allocation (debug: should show 4 GPUs)
nvidia-smi -L

# Launch one process per task/GPU (SLURM distributes across nodes if multi-node)
srun python -u main.py 

echo "Done"
