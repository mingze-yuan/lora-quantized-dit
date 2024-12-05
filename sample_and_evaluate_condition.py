# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
from torch import nn
import torch.distributed as dist
from models_lora import DiT_XL_2, DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from evaluator import *

import os
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '8888'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
time_condition_aware = False

# dist.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)


################# LoRA Model #################
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=128, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Low rank
        self.alpha = alpha  # Scaling factor

        # Low-rank matrices
        self.A = nn.Linear(original_layer.in_features, r, bias=False)
        self.B = nn.Linear(r, original_layer.out_features, bias=False)
        
        # Initialize weights of A and B as in the original implementation
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.B.weight, mean=0.0, std=0.01)

        # MLP to compute scale from the condition `c`
        self.scale_mlp = nn.Sequential(
            nn.Linear(1152, 12),
            nn.ReLU(),
            nn.Linear(12, 1),  # Outputs a scalar
            nn.Sigmoid()  # Keeps the scale in the range [0, 1]
        )

    def forward(self, x, c=None):
        lora_adjustment = self.B(self.A(x))  # (batch_size, in_features) -> (batch_size, out_features)

        # Compute the scale based on `c`
        if c is not None:
            scale = self.scale_mlp(c).squeeze(-1)  # Compute scalar for each batch element
            scale = scale.unsqueeze(1).unsqueeze(2)  # Match batch dimensions for broadcasting
        else:
            scale = self.alpha / self.r  # Default scale if no `c` is provided
            
        return self.original_layer(x) + scale * lora_adjustment


def add_lora_to_model(model, r=128, alpha=1.0):
    layers_to_modify = []  # Collect layers to modify first

    # Collect all linear layers in a list
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("mlp" in name or "attn" in name) and ("blocks" in name):
            layers_to_modify.append((name, module))

    # Replace each collected layer with a LoRA layer
    for name, module in layers_to_modify:
        # Split the name by '.' to traverse submodules and set the new layer correctly
        submodule = model
        *module_names, layer_name = name.split(".")
        for module_name in module_names:
            submodule = getattr(submodule, module_name)

        # Replace the layer with a LoRA layer
        # print(submodule)
        # print(layer_name)
        # print(module_name)
        setattr(submodule, layer_name, LoRALayer(module, r=r, alpha=alpha))
    model.time_condition_aware = time_condition_aware

# Assuming `model` is the DiT model with LoRA layers added
def freeze_model_weights(model):
    for name, param in model.named_parameters():
        if not ((".A" in name) or ('.B' in name) or ('.scale_mlp' in name)):  # Replace with the identifier for LoRA parameters
            param.requires_grad = False  # Freeze base model weights


def create_npz_from_sample_folder(sample_dir, num=5_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    print("Set up ddp")
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}."
    )

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size,
                                   num_classes=args.num_classes).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    # state_dict = find_model(ckpt_path)
   
    # Add LoRA layers (as shown in previous responses) and freeze base model weights
    add_lora_to_model(model, r=args.rk)  # Add LoRA layers to the model
    # freeze_model_weights(model)  # Freeze original model weights
    print("=" * 100)
    print(ckpt_path)
    print("=" * 100)
    if os.path.isfile(ckpt_path):
        print(f"Load the checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print("CANNOT FIND THE PATH!")
        raise Exception
        
    model.to(device)
    # model.time_condition_aware = True
        
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(
        ".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/dit_npz/{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(
        math.ceil(args.num_fid_samples / global_batch_size) *
        global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size(
    ) == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n,
                        model.in_channels,
                        latent_size,
                        latent_size,
                        device=device)
        y = torch.randint(0, args.num_classes, (n, ), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(sample_fn,
                                          z.shape,
                                          z,
                                          clip_denoised=False,
                                          model_kwargs=model_kwargs,
                                          progress=False,
                                          device=device)
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0,
                              255).permute(0, 2, 3,
                                           1).to("cpu",
                                                 dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(
                f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
        torch.cuda.empty_cache()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()
    
    # Download at https://github.com/openai/guided-diffusion/tree/main/evaluations
    ref_folder_dir = f"VIRTUAL_imagenet{args.image_size}"
    # create_npz_from_sample_folder(ref_folder_dir, args.num_fid_samples)
    evaluate(f"{ref_folder_dir}.npz", f"{sample_folder_dir}.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        choices=list(DiT_models.keys()),
                        default="DiT-XL/2")
    parser.add_argument("--vae",
                        type=str,
                        choices=["ema", "mse"],
                        default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=10_000)
    parser.add_argument("--image-size",
                        type=int,
                        choices=[256, 512],
                        required=True)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, required=True)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=
        "By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help=
        "Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model)."
    )
    
    parser.add_argument(
        "--rk",
        type=int,
        required=True,
        help=
        "rank of the model."
    )
    args = parser.parse_args()
    print("Number of GPUs",  torch.cuda.device_count())
    main(args)
