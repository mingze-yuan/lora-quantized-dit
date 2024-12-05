import torch
import torch.nn as nn
import re


# Define LoRA Layer
class LoRALayer(nn.Module):

    def __init__(self, original_layer, r=8, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Low rank
        self.alpha = alpha  # Scaling factor

        # Low-rank matrices
        self.A = nn.Parameter(
            torch.randn(original_layer.out_features, r) * 0.01)
        self.B = nn.Parameter(
            torch.randn(r, original_layer.in_features) * 0.01)

        # Scaling factor to ensure initial LoRA impact is small
        self.scale = alpha / self.r

    def forward(self, x):
        lora_adjustment = (
            x @ self.B.T
        ) @ self.A.T  # (batch_size, in_features) -> (batch_size, out_features)
        return self.original_layer(x) + lora_adjustment * self.scale


def add_lora_to_model(model, alpha=1.0):
    layers_to_modify = []  # Collect layers to modify first

    # Collect all linear layers in a list
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("mlp" in name or "attn" in name):
            layers_to_modify.append((name, module))

    # Replace each collected layer with a LoRA layer
    for name, module in layers_to_modify:
        match = re.compile(r"blocks\.(\d+)").search(name)
        if match:
            block_num = int(match.group(1))  # Extract block number as integer
            
            # Decide the LoRA rank based on the block number
            if "attn.proj" in name:
                # Attention proj layers' ranks: 16, 8, 8 for the 3 groups
                if block_num < 9:
                    r = 16
                else:
                    r = 8
            else:
                # MLP layers' ranks: 32, 16, 8 for the 3 groups
                if block_num < 9:
                    r = 32
                elif block_num < 18:
                    r = 16
                else:
                    r = 8
            # Split the name by '.' to traverse submodules and set the new layer correctly
            submodule = model
            *module_names, layer_name = name.split(".")
            for module_name in module_names:
                submodule = getattr(submodule, module_name)

            # Replace the layer with a LoRA layer
            setattr(submodule, layer_name, LoRALayer(module, r=r, alpha=alpha))


# Assuming `model` is the DiT model with LoRA layers added
def freeze_model_weights(model):
    for name, param in model.named_parameters():
        if not (
            (".A" in name) or
            ('.B' in name)):  # Replace with the identifier for LoRA parameters
            param.requires_grad = False  # Freeze base model weights
