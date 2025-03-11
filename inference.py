import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MNISTDataset
from diffusion.ddpm import DDPMScheduler
from models.unet import UnetModel
import argparse
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

tensor_to_pil = transforms.Compose([
     transforms.Lambda(lambda t: t*255),
     transforms.Lambda(lambda t: t.type(torch.uint8)),
     transforms.ToPILImage(),
 ])

def parse_args():
    parser = argparse.ArgumentParser(description="Test a mini diffusion model")

    # Model parameters
    parser.add_argument("--model_arch", type=str, default="unet", choices=["unet", "dit", "uvit"],
                        help="Architecture of the model to use. ")
    parser.add_argument("--in_channels", type=int, default=1, 
                        help="Number of input channels (e.g., 1 for grayscale images, 3 for RGB). Default is 1.")
    parser.add_argument("--image_size", type=int, default=64, 
                        help="Size of the input images (height and width). Default is 64.")
    parser.add_argument("--model_load_dir", type=str, default="/data/lixin/practice/diffusion_from_scratch/results/NNIST_train_20250310/model/model_epoch_900.pt",
                    help="Directory to load the model weights from")
    parser.add_argument("--output_dir", type=str, default="/data/lixin/output",
                        help="Directory to save the output images")
    parser.add_argument("--num_images", type=int, default=20, help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of images to generate")

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
    
    return args

def save_combined_image(tensors, path, grid_size, img_size):
    """
    将一组张量图像汇总到一个大图像中并保存。
    
    Args:
        tensors (list of torch.Tensor): 要汇总的张量列表。
        path (str): 保存图像的路径。
        grid_size (tuple): (行数, 列数)。
        img_size (tuple): 每个小图像的大小 (宽, 高)。
    """
    rows, cols = grid_size
    width, height = img_size
    combined_image = Image.new('RGB', (cols * width, rows * height))

    for idx, tensor in enumerate(tensors):
        img = tensor_to_pil(tensor)
        row = idx // cols
        col = idx % cols
        combined_image.paste(img, (col * width, row * height))

    combined_image.save(path)

if __name__ == "__main__":
    args = parse_args()

    if args.model_arch == "unet":
        model = UnetModel(args.in_channels).to(args.device)
    elif args.model_arch == "dit":
        pass
    elif args.model_arch == "uvit":
        pass
    else:
        raise ValueError(f"unexpected model architecture for {args.model_arch}")
    
    if not os.path.exists(args.model_load_dir):
        raise FileNotFoundError(f"the specified path does not exists")
    
    model_weights = torch.load(args.model_load_dir)
    missing_keys, unexcepted_keys = model.load_state_dict(model_weights, strict = True)
    print(f"succeed to load model weights with {len(missing_keys)} missing keys and {len(unexcepted_keys)} unexcepted keys")

    diffusion_scheduler = DDPMScheduler(args.beta_schedule, args.noise_steps, args.beta_start, args.beta_end, args.cosine_s)
    
    batch_size = args.batch_size
    image_size = args.image_size
    num_imgs = args.num_images
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    batch_x_t = torch.randn(size=(batch_size, 1, image_size, image_size)).to(args.device)
    batch_cls = torch.arange(start = 0, end = 10, dtype = torch.long).to(args.device)
    denoised_images = diffusion_scheduler.denoise_sample(model, batch_x_t, batch_cls)

    # 保存每个批次的汇总图像
    combined_images = []
    for b in range(batch_size):
        img_list = []
        for i in range(0, num_imgs):
            idx = int(1000 / num_imgs) * (i + 1)
            # 像素值还原到 [0, 1]
            final_img = (denoised_images[idx][b].to('cpu') + 1) / 2
            img_list.append(final_img)

        # 保存汇总图像
        grid_size = (1, num_imgs)
        img_path = os.path.join(output_dir, f"batch_{b}_combined.png")
        save_combined_image(img_list, img_path, grid_size, (image_size, image_size))
        combined_images.append(img_path)

    # 拼接所有汇总图像
    final_image_width = image_size * num_imgs
    final_image_height = image_size * batch_size
    final_image = Image.new('RGB', (final_image_width, final_image_height))

    for i, img_path in enumerate(combined_images):
        img = Image.open(img_path)
        final_image.paste(img, (0, i * image_size))

    # 保存最终的大图像
    final_image_path = os.path.join(output_dir, "final_combined_image.png")
    final_image.save(final_image_path)