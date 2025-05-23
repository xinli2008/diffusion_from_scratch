import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import CustomDataset
from diffusion.ddpm import DDPMScheduler
from models.unet import UnetModel
from models.dit import DiT_models
from models.uvit import Uvit_models
import argparse
from tqdm import tqdm

def get_dataset_channels(dataset_name):
    channels = {
        "MNIST": 1,
        "CIFAR-10": 3,
    }
    return channels.get(dataset_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a mini diffusion model on custom dataset")

    # Training parameters
    parser.add_argument("--max_epoch", type=int, default=2000, 
                        help="Maximum number of epochs to train the model. Default is 50.")
    parser.add_argument("--learning_rate", type=float, default=0.0005, 
                        help="Learning rate for the optimizer. Default is 0.001.")
    parser.add_argument("--batch_size", type=int, default=512, 
                        help="Number of samples per batch during training. Default is 64.")
    parser.add_argument("--result_dir", type=str, default="results/CIFAR_train_20250330", 
                        help="Directory to save training results, including models and logs. Default is 'results'.")
    parser.add_argument("--model_save_interval", type=int, default=100, 
                        help="Interval (in epochs) at which to save the model. Default is 10.")
    # Model parameters
    parser.add_argument("--model_arch", type=str, default="dit", choices=["unet", "dit", "uvit"],
                        help="Architecture of the model to use. ")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, default="CIFAR-10", choices=["MNIST", "CIFAR-10"], 
                        help="Name of the dataset to use. Default is 'MNIST'.")
    parser.add_argument("--dataset_path", type=str, default="./data", 
                        help="Path where the dataset will be stored or is already located. Default is './data'.")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="Number of worker threads for data loading. Default is 8.")
    parser.add_argument("--image_size", type=int, default=32, 
                        help="Size of the input images (height and width). Default is 64.")

    # Diffusion parameters
    parser.add_argument("--noise_steps", type=int, default=1000, 
                        help="Number of timesteps in the diffusion process. Default is 1000.")
    parser.add_argument("--beta_schedule", type=str, choices=["linear", "cosine", "sqrt_linear"],
                        default="linear", help = "Type of beta schedule to use. Options are 'linear', 'cosine' and 'sqrt_linear'. Default is 'linear'.")
    parser.add_argument("--beta_start", type=float, default=1e-4, 
                        help="Starting value of linear beta schedule. Default is 1e-4.")
    parser.add_argument("--beta_end", type=float, default=2e-2, 
                        help="Ending value of linear beta schedule. Default is 2e-2.")
    parser.add_argument("--cosine_s", type=float, default=8e-3, 
                        help="Parameter s used in cosine beta schedule. Default is 8e-3.")

    # Others
    parser.add_argument("--device", type=str, default="cuda:0", 
                        help="Device to use for training (e.g., 'cuda:0' for GPU or 'cpu'). Default is 'cuda:0'.")

    # Parse arguments
    args = parser.parse_args()

    # Create directories for saving models and logs
    args.model_save_dir = os.path.join(args.result_dir, "model")
    args.log_save_dir = os.path.join(args.result_dir, "logs")
    
    return args

def train_one_epoch(model, dataloader, diffusion_scheduler, timestep, optimizer, epoch, device, writer):
    model.train()
    epoch_loss = 0

    dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for step, (batch_x, batch_cls) in enumerate(dataloader_tqdm):
        batch_x = batch_x.to(device)
        batch_cls = batch_cls.to(device)

        # Sample a random timestep for each image
        batch_t = torch.randint(low=0, high=timestep, size=(batch_x.shape[0],)).to(device)        

        # Add noise to the image according to the noise magnitude at each timestep
        batch_x_t, batch_noise_t = diffusion_scheduler.q_sample(batch_x, batch_t)
        
        # Predict the noise residual and compute loss
        batch_predict_t = model(batch_x_t, batch_t, batch_cls)

        loss = F.mse_loss(batch_predict_t, batch_noise_t, reduction="mean")     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    return avg_loss

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.log_save_dir):
        os.makedirs(args.log_save_dir)
    if not os.path.exists(args.dataset_path):
        os.makedirs(args.dataset_path)
    writer = SummaryWriter(log_dir=args.log_save_dir)

    in_channels = get_dataset_channels(args.dataset_name)
    if in_channels is None:
        raise ValueError(f"Unexpected dataset type: {args.dataset_name}")
    
    if args.model_arch == "unet":
        model = UnetModel(in_channels).to(args.device)
    elif args.model_arch == "dit":
        model = DiT_models["DiT-B/8"](image_size = (args.image_size, args.image_size), input_channel = in_channels, num_labels = 10).to(args.device)
    elif args.model_arch == "uvit":
        model = Uvit_models["UVit_base"](img_size = (args.image_size, args.image_size), in_channels = in_channels, num_classes = 10).to(args.device)
    else:
        raise ValueError(f"unexpected model architecture for {args.model_arch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_dataset = CustomDataset(root=args.dataset_path, dataset_type = args.dataset_name, train=True, image_size = args.image_size)
    
    diffusion_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    diffusion_scheduler = DDPMScheduler(args.beta_schedule, args.noise_steps, args.beta_start, args.beta_end, args.cosine_s)

    for epoch in range(args.max_epoch):
        print(f"Start training at epoch {epoch}")
        avg_loss = train_one_epoch(model, diffusion_dataloader, diffusion_scheduler, args.noise_steps, optimizer, epoch, args.device, writer)
        print(f"End training at epoch {epoch}, Average Loss = {avg_loss:.4f}")

        # Save model
        if epoch % args.model_save_interval == 0 and epoch!=0:
            model_save_path = os.path.join(args.model_save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    writer.close()