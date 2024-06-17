import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import *
from dataset import train_dataset
from diffusion import forward_diffusion
from time_position_emb import TimePositionEmbedding
from unet import UNet

# 创建数据加载器
diffusion_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 初始化模型
model = UNet(1).to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数
loss_fn = nn.L1Loss()

def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch, device, writer):
    """
    训练模型一个周期。

    Args:
        model (nn.Module): 需要训练的模型。
        dataloader (DataLoader): 训练数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        loss_fn (nn.Module): 损失函数。
        epoch (int): 当前训练的周期数。
        device (torch.device): 训练设备。
        writer (SummaryWriter): TensorBoard的SummaryWriter实例。

    Returns:
        float: 当前周期的平均损失。
    """
    model.train()
    epoch_loss = 0

    for step, (batch_x, batch_cls) in enumerate(dataloader):
        batch_x = batch_x.to(device) * 2 - 1

        # timestep
        batch_t = torch.randint(low=0, high=timestep, size=(batch_x.shape[0],)).to(device)

        # classification information
        batch_cls = batch_cls.to(device)

        # 前向扩散过程
        batch_x_t, batch_noise_t = forward_diffusion(batch_x, batch_t)

        # 预测
        batch_predict_t = model(batch_x_t, batch_t, batch_cls)

        # 计算损失
        loss = loss_fn(batch_predict_t, batch_noise_t)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 将损失记录到 TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

    return avg_loss

if __name__ == "__main__":
    # 创建目录以保存模型和日志文件
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 初始化 TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir='logs')

    for epoch in range(max_epoch):
        print(f"Start training at epoch {epoch}")
        avg_loss = train_one_epoch(model, diffusion_dataloader, optimizer, loss_fn, epoch, device, writer)
        print(f"End training at epoch {epoch}, Average Loss = {avg_loss:.4f}")

        # 保存模型
        if epoch % 50 == 0:
            model_save_path = os.path.join('models', f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    writer.close()
