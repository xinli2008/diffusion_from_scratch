import torch
from config import *
from dataset import train_dataet, tensor_to_pil
from matplotlib import pyplot as plt

# diffusion的前向传播中需要的参数
betas = torch.linspace(start = 0.0001, end = 0.02, steps = timestep)  # [timestep]
alpha = 1 - betas

# 累乘的含义:
# 给定一个序列[a1, a2, a3, ..., an], 累乘的结果应该是: [a1, a1*a2, a1*a2*a3, ....]
alpha_cum_product = torch.cumprod(alpha, dim = 0)  

def forward_diffusion(batch_x, batch_t):
    r"""
    Function: 执行前向加噪的过程
    batch_x: [b, c, w, h]
    batch_t: [b,]
    """
    # 为每张图片生成第t个时刻的高斯噪声, torch.randn可以生成符合标准正态分布的数据。
    batch_noise_t = torch.randn_like(batch_x) 
    # reshape
    batch_alpha_cum_product = alpha_cum_product.to(device)[batch_t].view(batch_x.shape[0], 1, 1, 1)
    # 利用diffusion前向传播中的X_t公式, 直接计算x0在t时刻加上噪声后得到的结果
    # Q: 这里为什么可以相乘呢?
    # A:在计算 torch.sqrt(batch_alpha_cum_product) * batch_x 时：batch_alpha_cum_product 会广播为 [b, c, w, h]，与 batch_x 的形状相同。
    batch_x_t = torch.sqrt(batch_alpha_cum_product) * batch_x + torch.sqrt(1 - batch_alpha_cum_product) * batch_noise_t
    return batch_x_t, batch_noise_t

if __name__ == "__main__":
    # [2, 1, 48, 48]
    batch_x = torch.stack((train_dataet[0][0], train_dataet[9][0]), dim = 0).to(device)

    # 加噪前的样子
    input_image_0 = tensor_to_pil(batch_x[0])
    input_image_1 = tensor_to_pil(batch_x[1])
    input_image_0.save("/mnt/diffusion_from_scratch/saved_debug_images/input_image_0.jpg")
    input_image_0.save("/mnt/diffusion_from_scratch/saved_debug_images/input_image_1.jpg")

    # 本来像素在[0,1]之间,将像素值调整到[-1,1]之间,方便与高斯噪声匹配
    batch_x = batch_x * 2 -1 
    batch_t = torch.randint(0, timestep, size = (batch_x.shape[0], )).to(device)
    print('batch_t:',batch_t)

    batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)
    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    # 加噪后的样子
    output_image_0 = tensor_to_pil((batch_x_t[0]+1)/2)
    output_image_1 = tensor_to_pil((batch_x_t[1]+1)/2)
    output_image_0.save("/mnt/diffusion_from_scratch/saved_debug_images/output_image_0.jpg")
    output_image_1.save("/mnt/diffusion_from_scratch/saved_debug_images/output_image_1.jpg")
