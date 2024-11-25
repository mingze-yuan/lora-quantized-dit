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



if __name__ == '__main__':
    image_size = 256
    device = 'cuda'
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    checkpoint_dir = '/n/netscratch/nali_lab_seas/Everyone/mingze/datasets/lora_training_w8a8/checkpoints'
    # Load model:
    latent_size = int(image_size) // 8
    selected_class_ids = [0, 1, 2, 3]
    seed = 1 #@param {type:"number"}
    torch.manual_seed(seed)
    num_sampling_steps = 100 #@param {type:"slider", min:0, max:1000, step:1}
    cfg_scale = 2 #@param {type:"slider", min:1, max:10, step:0.1}
    class_labels = selected_class_ids #@param {type:"raw"}
    samples_per_row = 4 #@param {type:"number"}
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])    

    full_dataset = ImageFolder("/n/home11/mingzeyuan/.cache/kagglehub/datasets/ifigotin/imagenetmini-1000/versions/1/imagenet-mini/train", transform=transform)
    # selected_indices = []
    # for i, (_, label) in tqdm(enumerate(full_dataset)):
    #     if label in selected_class_ids:
    #         selected_indices.append(i)
    #     if label > max(selected_class_ids):
    #         break
        
    def init_models():
        model = DiT_models['DiT-XL/2'](input_size=latent_size).to(device)
        # state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
        state_dict = torch.load('/n/netscratch/nali_lab_seas/Everyone/mingze/models/pretrained_models/DiT-XL-2-256x256.pt', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval() # important!
        vae = AutoencoderKL.from_pretrained(vae_model).to(device)

        return model, vae
    
    epochs = 200
    learning_rate = 1e-4
    batch_size = 64
    
    model, vae = init_models()

    model = quantize_dit(model, mode='W8A8')
    add_lora_to_model(model)  # Add LoRA layers to the model
    freeze_model_weights(model)  # Freeze original model weights
    model.to(device)
    # filtered_dataset = Subset(full_dataset, selected_indices)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    diffusion = create_diffusion(str(num_sampling_steps))

    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)

            # Encode images to latent space and normalize latents
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)

            # Explicitly set requires_grad for the input latents
            # Sample a random timestep for each batch
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = {"y": y}

            # Compute training losses from diffusion
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
        
        checkpoint = {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
            }
        
        checkpoint_path = f"{checkpoint_dir}/latest.pt"
        torch.save(checkpoint, checkpoint_path)
        # Create sampling noise:
        
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
        save_image(samples, f"/n/netscratch/nali_lab_seas/Everyone/mingze/datasets/lora_training_w8a8/samples/sample_{epoch:03d}.png", nrow=int(samples_per_row), normalize=True, value_range=(-1, 1))

    print("Training completed.")
    
                
    
            
    
    
    




