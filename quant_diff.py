import torch
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
from quant.fake_quant import quantize_dit
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from quant.lora_utils import add_lora_to_model, freeze_model_weights
from torch import optim
from train import center_crop_arr
from collections import OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compute_lora_rank_suggestions(model,
                                  original_state_dict,
                                  quantized_state_dict,
                                  energy_threshold=0.95):
    """
    Computes the suggested LoRA rank for each linear layer based on the difference between 
    original and quantized weights.

    Args:
        model (nn.Module): The model to analyze.
        original_state_dict (dict): State dict with original weights.
        quantized_state_dict (dict): State dict with quantized weights.
        energy_threshold (float): The percentage of energy to retain (default: 0.95).

    Returns:
        dict: A mapping from layer names to suggested LoRA ranks.
    """
    rank_suggestions = OrderedDict()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ("mlp" in name or "attn" in name):
            # Extract original and quantized weights
            original_weight = original_state_dict[f"{name}.weight"].to(device)
            quantized_weight = quantized_state_dict[f"{name}.weight"].to(device)

            # Compute the difference
            weight_diff = original_weight - quantized_weight

            # Perform SVD on the difference
            u, s, v = torch.svd(weight_diff)

            # Compute the total energy
            total_energy = torch.sum(s**2).item()

            # Determine the minimum rank that retains the energy threshold
            cumulative_energy = torch.cumsum(s**2, dim=0)
            rank = torch.sum(cumulative_energy /
                             total_energy < energy_threshold).item() + 1

            # Print name and suggested rank
            print(f"Layer name: {name}, Shape: {weight_diff.shape}, Suggested rank: {rank}")


if __name__ == '__main__':
    image_size = 256
    checkpoint_dir = '/n/netscratch/nali_lab_seas/Everyone/mingze/datasets/lora_training_w8a8/checkpoints'

    latent_size = int(image_size) // 8
    seed = 1
    torch.manual_seed(seed)

    model = DiT_models['DiT-XL/2'](input_size=latent_size).to(device)
    original_state_dict = torch.load(
        '/n/netscratch/nali_lab_seas/Everyone/mingze/models/pretrained_models/DiT-XL-2-256x256.pt',
        weights_only=True)
    model.load_state_dict(original_state_dict)
    model.eval()  # important!

    model = quantize_dit(model, mode='W8A8')
    quantized_state_dict = model.state_dict()

    # Compute suggested LoRA ranks for each linear layer
    compute_lora_rank_suggestions(model, original_state_dict,
                                                    quantized_state_dict)

"""
Layer name: t_embedder.mlp.0, Shape: torch.Size([1152, 256]), Suggested rank: 188
Layer name: t_embedder.mlp.2, Shape: torch.Size([1152, 1152]), Suggested rank: 400
Layer name: blocks.0.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.0.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.0.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.0.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 989
Layer name: blocks.1.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.1.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.1.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.1.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 989
Layer name: blocks.2.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.2.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.2.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.2.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.3.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.3.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.3.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.3.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.4.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.4.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.4.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.4.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.5.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.5.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.5.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.5.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.6.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.6.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.6.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.6.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.7.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.7.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.7.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.7.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.8.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.8.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.8.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.8.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.9.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.9.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.9.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.9.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.10.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.10.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.10.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.10.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.11.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.11.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.11.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.11.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.12.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.12.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.12.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.12.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.13.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.13.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.13.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.13.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.14.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.14.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.14.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.14.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.15.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.15.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.15.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.15.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.16.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.16.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.16.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.16.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.17.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.17.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.17.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.17.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.18.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.18.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.18.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.18.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.19.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.19.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.19.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.19.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.20.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.20.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.20.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.20.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.21.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.21.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.21.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.21.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.22.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.22.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.22.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.22.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.23.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.23.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.23.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.23.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.24.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.24.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 703
Layer name: blocks.24.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.24.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.25.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.25.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.25.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.25.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 991
Layer name: blocks.26.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 960
Layer name: blocks.26.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.26.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.26.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
Layer name: blocks.27.attn.qkv, Shape: torch.Size([3456, 1152]), Suggested rank: 961
Layer name: blocks.27.attn.proj, Shape: torch.Size([1152, 1152]), Suggested rank: 702
Layer name: blocks.27.mlp.fc1, Shape: torch.Size([4608, 1152]), Suggested rank: 992
Layer name: blocks.27.mlp.fc2, Shape: torch.Size([1152, 4608]), Suggested rank: 992
"""