import torch 
from config import *
from diffusion import *
from dataset import tensor_to_pil
from unet import UNet
from PIL import Image
import os

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

def backward_denoise(model,batch_x_t):
    steps=[batch_x_t,]

    global alpha,alpha_cum_product,variance

    model=model.to(device)
    batch_x_t=batch_x_t.to(device)
    alpha=alpha.to(device)
    alpha_cum_product=alpha_cum_product.to(device)
    variance=variance.to(device)
    
    # 应该是由于BN层mean,std导致的eval效果不好,先不开启了
    #model.eval()
    with torch.no_grad():
        for t in range(timestep-1,-1,-1):
            batch_t=torch.full((batch_x_t.size(0),),t).to(device) #[999,999,....]
            # 预测x_t时刻的噪音
            batch_predict_noise_t=model(batch_x_t,batch_t)
            # 生成t-1时刻的图像
            shape=(batch_x_t.size(0),1,1,1)
            batch_mean_t=1/torch.sqrt(alpha[batch_t].view(*shape))*  \
                (
                    batch_x_t-
                    (1-alpha[batch_t].view(*shape))/torch.sqrt(1-alpha_cum_product[batch_t].view(*shape))*batch_predict_noise_t
                )
            if t!=0:
                batch_x_t=batch_mean_t+ \
                    torch.randn_like(batch_x_t)* \
                    torch.sqrt(variance[batch_t].view(*shape))
            else:
                batch_x_t=batch_mean_t
            batch_x_t=torch.clamp(batch_x_t, -1.0, 1.0).detach()
            steps.append(batch_x_t)
    return steps 

if __name__ == '__main__':
    # 加载状态字典
    state_dict = torch.load("/mnt/diffusion_from_scratch/models/model_epoch_190.pt")
    model = UNet(1)

    # 加载状态字典到模型
    model.load_state_dict(state_dict)

    # 生成噪音图
    batch_size = 10
    image_size = 48  # 确保定义 image_size
    batch_x_t = torch.randn(size=(batch_size, 1, image_size, image_size))  # (10, 1, 48, 48)

    # 逐步去噪得到原图
    steps = backward_denoise(model, batch_x_t)

    # 创建输出目录
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # 绘制数量
    num_imgs = 20

    # 保存每个批次的汇总图像
    combined_images = []
    for b in range(batch_size):
        img_list = []
        for i in range(0, num_imgs):
            idx = int(timestep / num_imgs) * (i + 1)
            # 像素值还原到 [0, 1]
            final_img = (steps[idx][b].to('cpu') + 1) / 2
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