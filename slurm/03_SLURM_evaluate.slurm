#!/bin/bash
#SBATCH -c 16               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_compute         # Partition to submit to
#SBATCH --mem=16G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=mingzeyuan@g.harvard.edu

cd ~
source .bashrc
mamba activate qdit
cd DiT/

# Check SLURM GPU allocation
echo "SLURM job ID: $SLURM_JOB_ID"

# Run your training script
python evaluator.py /n/netscratch/nali_lab_seas/Everyone/mingze/datasets/dit_samples/VIRTUAL_imagenet256_labeled.npz /n/netscratch/nali_lab_seas/Everyone/mingze/datasets/dit_samples/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-ema-cfg-1.5-seed-0.npz