#!/bin/bash
#SBATCH -c 16               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue         # Partition to submit to
#SBATCH --mem=16G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1        # Request 4 GPUs
#SBATCH -o logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

cd ~
source .bashrc
mamba activate qdit
cd DiT/

# Check SLURM GPU allocation
echo "SLURM job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Check NVIDIA GPU status
nvidia-smi || echo "nvidia-smi not available"

# Verify CUDA availability in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

python train_lora.py