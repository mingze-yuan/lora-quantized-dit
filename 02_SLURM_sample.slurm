#!/bin/bash
#SBATCH -c 64               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu        # Partition to submit to
#SBATCH --mem=64G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4        # Request 4 GPUs
#SBATCH -o logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --exclusive

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

# Run your training script
torchrun --nnodes=1 \
    --nproc_per_node=4 sample_ddp.py \
    --model DiT-XL/2 \
    --num-fid-samples 50000 \
    --ckpt $DATA/models/pretrained_models/DiT-XL-2-256x256.pt \
    --sample-dir $DATA/datasets/dit_samples_tmp