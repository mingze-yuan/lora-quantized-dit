import torch
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
from torchvision.utils import save_image
from quant.fake_quant import quantize_dit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm.auto import tqdm
from quant.lora_utils import add_lora_to_model, freeze_model_weights
from torch import optim
from train import center_crop_arr
from collections import OrderedDict


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
        if isinstance(module, torch.nn.Linear):
            # Extract original and quantized weights
            original_weight = original_state_dict[f"{name}.weight"]
            quantized_weight = quantized_state_dict[f"{name}.weight"]

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

            # Store the suggested rank
            rank_suggestions[name] = rank

    return rank_suggestions


if __name__ == '__main__':
    image_size = 256
    device = 'cuda'
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
    suggested_ranks = compute_lora_rank_suggestions(model, original_state_dict,
                                                    quantized_state_dict)

    # Print the suggestions
    for layer_name, rank in suggested_ranks.items():
        print(f"Layer: {layer_name}, Suggested LoRA Rank: {rank}")
