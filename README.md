## Advancing Accuracy-Recovery Adapters for Quantized Diffusion Transformers
Author: Mingze Yuan, Xingyu Xiang, Yida Chen, Golden Chang
Instructor: H.T. Kang
Class: CS 2420

## Setup

We provide an [`environment_lora`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment_lora
conda activate dit
```

## Finetuning DiT with CA-LoRA

We provide a training script for DiT in [`condition_aware_lora`](train.py). This script can be used to train condition aware LoRA on an absmax quantized DiT model.

You can specifiy the bit depth of the quantized DiT model by modifying the global variable `bit_depth = "{select one number from 4, 6, 7, or 8}"`. The global rank of the LoRA model can be changed using `rank = {an integer number}` variable. `condition_aware` variable controls whether the condition awareness is used in the finetuning. Set this variable to `condition_aware = ""` for condition-aware LoRA. Set it to `condition_aware="no"` for the original LoRA finetuning. `layer_aware` variable controls whether variable rank will be used for MLP and attention layers. Set this variable to `layer_aware="_yes"` for cross-layer variable rank. Set it to `layer_aware=""` for a single global rank that is applied to all LoRA modules.


## Evaluating Finetuned CA-LoRA + DiT
To evaluate the finetune model, you can run `run_all_condition.sh` script. Make sure you change the filepath to your saved finetuned weights and the rank of the finetuned LoRA model accordingly before you run this code. The results will be saved to a folder named `results/`. 


## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.
