## Scalable Diffusion Models with Transformers (DiT)<br><sub>Official PyTorch Implementation</sub>

## Setup


We provide an [`environment_lora`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment_lora
conda activate dit
```

## Finetuning DiT with CA-LoRA

We provide a training script for DiT in [`condition_aware_lora`](train.py). This script can be used to train condition aware lora on an absmax quantized DiT model




## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.
