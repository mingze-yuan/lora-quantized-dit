#!/bin/bash

steps=(50 100 250)
sizes=(256 512)
models=(
    "w4a8_absmax.pth"
    "w4a8_sq.pth"
    "w4a8_qdit.pth"
    "w4a8_ptq4dit.pth"
    "w4a8_lora.pth"
    "w8a8_absmax.pth"
    "w8a8_sq.pth"
    "w8a8_qdit.pth"
    "w8a8_ptq4dit.pth"
    "w8a8_lora.pth"
)

mkdir -p results

for step in "${steps[@]}"; do
    for size in "${sizes[@]}"; do
        for model in "${models[@]}"; do
            python sample_and_evaluate.py --num-sampling-steps "$step" --image-size "$size" --ckpt "checkpoints/$model" > "results/$model-$size-$step.txt"
        done
    done
done