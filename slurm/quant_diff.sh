#!/bin/bash
#SBATCH -c 16               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue         # Partition to submit to
#SBATCH --mem=64G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1        # Request 4 GPUs
#SBATCH -o logs/quant_diff_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/quant_diff_%j.err  # File to which STDERR will be written, %j inserts jobid

#salloc -c 16 -t 0-12:00 -p gpu_requeue --mem=64G --gres=gpu:nvidia_a100-sxm4-80gb:1

cd ~
cd lora-quantized-dit

# Check SLURM GPU allocation
echo "SLURM job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# Check NVIDIA GPU status
nvidia-smi || echo "nvidia-smi not available"

# Load Python environment
module load python/3.12.5-fasrc01
mamba activate DiT

# Verify CUDA availability in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

python quant_diff.py