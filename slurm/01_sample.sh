torchrun --nnodes=1 \
    --nproc_per_node=1 sample_ddp.py \
    --model DiT-XL/2 \
    --num-fid-samples 50000 \
    --ckpt $DATA/models/pretrained_models/DiT-XL-2-256x256.pt \
    --sample-dir $DATA/datasets/dit_samples