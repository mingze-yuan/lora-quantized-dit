import os
import torch
import torch.nn as nn
from torchvision.utils import save_image

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusers import DiTPipeline


from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

from models_lora import DiT_models  # Ensure this imports correctly

from download import find_model
from models_lora import DiT_XL_2
from PIL import Image
from IPython.display import display
# torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

    
################ Training Hyperparameter #################
bit_depth = "6"
rank = 128
batch_size = 64
learning_rate = 1e-3
epochs = 8

condition_aware = ""
layer_aware = "_yes"
time_condition_aware = False if condition_aware == "no" else True

image_size = 256 #@param [256, 512]
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
save_directory = "/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/"
latent_size = int(image_size) // 8
# Load model:
model = DiT_XL_2(input_size=latent_size).to(device)
state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
model.load_state_dict(state_dict)
# model.eval() # important!
vae = AutoencoderKL.from_pretrained(vae_model, cache_dir=save_directory).to(device)

################# Inference Hyperparameter #################
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 100 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 3 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = 20, 80, 124, 207, 360, 387, 567, 762 #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}
# Create diffusion object:
diffusion = create_diffusion(str(num_sampling_steps))

################# Quantization Function #################
# Function to quantize model weights to 8 bits and then cast them back to 32 bits
def quantize_to_8bit_and_back(model):
    for name, param in model.named_parameters():
        # Quantize to 8-bit (using scale and zero-point, typical for int8 quantization)
        scale = param.abs().max() / 127  # Scale based on max absolute value
        param_8bit = torch.clamp((param / scale).round(), -128, 127).to(torch.int8)

        # Dequantize to float32 using the 8-bit values
        param.data = (param_8bit.float() * scale).to(torch.float32)
        
        
def quantize_to_7bit_and_back(model):
    for name, param in model.named_parameters():
        if param.requires_grad:  # Ensure that only trainable parameters are quantized
            # Quantize to 7-bit (scale and zero-point typical for quantization)
            scale = param.abs().max() / 63  # Scale based on max absolute value for 7-bit range [-64, 63]
            param_7bit = torch.clamp((param / scale).round(), -64, 63).to(torch.int8)

            # Dequantize to float32 using the 6-bit values
            param.data = (param_7bit.float() * scale).to(torch.float32)
            
            
def quantize_to_6bit_and_back(model):
    for name, param in model.named_parameters():
        if param.requires_grad:  # Ensure that only trainable parameters are quantized
            # Quantize to 6-bit (scale and zero-point typical for quantization)
            scale = param.abs().max() / 31  # Scale based on max absolute value for 6-bit range [-32, 31]
            param_6bit = torch.clamp((param / scale).round(), -32, 31).to(torch.int8)

            # Dequantize to float32 using the 6-bit values
            param.data = (param_6bit.float() * scale).to(torch.float32)
            
        
# Function to quantize model weights to 4 bits and then cast them back to 32 bits
def quantize_to_4bit_and_back(model):
    for name, param in model.named_parameters():
        if param.requires_grad:  # Ensure that only trainable parameters are quantized
            # Quantize to 4-bit (scale and zero-point typical for quantization)
            scale = param.abs().max() / 7  # Scale based on max absolute value for 4-bit range [-8, 7]
            param_4bit = torch.clamp((param / scale).round(), -8, 7).to(torch.int8)

            # Dequantize to float32 using the 4-bit values
            param.data = (param_4bit.float() * scale).to(torch.float32)
            
        
################# Quantization rate #################
if bit_depth == "4":
    quantize_to_4bit_and_back(model)
elif bit_depth == "6":
    quantize_to_6bit_and_back(model)
elif bit_depth == "7":
    quantize_to_7bit_and_back(model)
elif bit_depth == "8":
    quantize_to_7bit_and_back(model)
    
print(f"Quantize the model to {bit_depth} bits")
    
model.eval()
torch.manual_seed(123)
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False,
    model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

epoch = 0
# Save and display images:
save_image(samples, f"temp_result{condition_aware}/{rank}_r_{bit_depth}_bit/original_sample_{epoch}_{rank}_r_{bit_depth}_bit{layer_aware}.png", nrow=int(samples_per_row),
           normalize=True, value_range=(-1, 1))
    
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
        if layer_aware == "_yes":
            if "attn.proj" in name:
                adjusted_r = int(r * 0.7)
            elif "attn.qkv" in name:
                adjusted_r = int(r * 0.9)
            else:
                adjusted_r = r
        else:
            adjusted_r = r
        # Replace the layer with a LoRA layer
        # print(submodule)
        # print(layer_name)
        # print(module_name)
        setattr(submodule, layer_name, LoRALayer(module, r=adjusted_r, alpha=alpha))
    model.time_condition_aware = time_condition_aware

# Assuming `model` is the DiT model with LoRA layers added
def freeze_model_weights(model):
    for name, param in model.named_parameters():
        if not ((".A" in name) or ('.B' in name) or ('.scale_mlp' in name)):  # Replace with the identifier for LoRA parameters
            param.requires_grad = False  # Freeze base model weights

# Add LoRA layers (as shown in previous responses) and freeze base model weights
add_lora_to_model(model, r=rank)  # Add LoRA layers to the model
freeze_model_weights(model)  # Freeze original model weights
model.to("cuda");


################# Load Dataset #################
# Dataset and Dataloader
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


full_dataset = ImageFolder("/n/holylabs/LABS/wattenberg_lab/Everyone/imagenet-medium/train_100000/train_100000/", transform=transform)
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


################# Training #################
# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

ckpt_path = f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/weights_backup/dit_{bit_depth}bit_{rank}r{condition_aware}{layer_aware}.pth"
if os.path.isfile(ckpt_path):
    print(f"Load the checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    del checkpoint
    torch.cuda.empty_cache()
else:
    start_epoch = 0
    
    
# loss = checkpoint['loss']
all_loss = []
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
for epoch in tqdm(range(start_epoch, epochs)):
    # Create sampling noise:
    if epoch != 0:
        model.eval()
        # torch.manual_seed(0)
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:

        save_image(samples, f"temp_result{condition_aware}/{rank}_r_{bit_depth}_bit/sample_{epoch}_{rank}_r_{bit_depth}_bit{layer_aware}.png", nrow=int(samples_per_row),
                   normalize=True, value_range=(-1, 1))
    
    model.train()
    running_loss = 0.0
    counter = 0
    for x, y in tqdm(dataloader):
        # Backward and optimize
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        # Encode images to latent space and normalize latents
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        # Sample a random timestep for each batch
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        # y.requires_grad_(True)
        model_kwargs = {"y": y}

        # Compute training losses from diffusion
        # t.requires_grad_(True)
        x = torch.Tensor(x)
        x.requires_grad_(True)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        loss.backward()
        optimizer.step()

        # Log loss
        running_loss += loss.item()
        counter += 1
        if counter % 128 == 0:
            print(f"Intermediate Results: Epoch [{epoch+1}/{epochs}], Loss: {running_loss/counter:.4f}")
        
        all_loss.append(running_loss/counter)
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': all_loss,
            }, f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/weights_backup/dit_{bit_depth}bit_{rank}r{condition_aware}{layer_aware}.pth")
    
    
epoch += 1
model.eval()
# torch.manual_seed(0)
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False,
    model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Save and display images:

save_image(samples, f"temp_result{condition_aware}/{rank}_r_{bit_depth}_bit/sample_{epoch}_{rank}_r_{bit_depth}_bit{layer_aware}.png", nrow=int(samples_per_row),
           normalize=True, value_range=(-1, 1))
print("Training completed.")
