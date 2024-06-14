import torch 
from torch import nn 
from dataset import train_dataset
from config import * 
from diffusion import forward_diffusion
from time_position_emb import TimePositionEmbedding
from conv_block import ConvBlock

class UNet(nn.Module):
    def __init__(self,img_channel,channels=[64, 128, 256, 512, 1024],time_emb_size=256):
        super().__init__()

        channels=[img_channel]+channels
        
        # time转embedding
        self.time_emb=nn.Sequential(
            TimePositionEmbedding(time_emb_size),
            nn.Linear(time_emb_size,time_emb_size),
            nn.ReLU(),
        )

        # 每个encoder conv block增加一倍通道数
        self.enc_convs=nn.ModuleList()
        for i in range(len(channels)-1):
            self.enc_convs.append(ConvBlock(channels[i],channels[i+1],time_emb_size))
        
        # 每个encoder conv后马上缩小一倍图像尺寸,最后一个conv后不缩小
        self.maxpools=nn.ModuleList()
        for i in range(len(channels)-2):
            """
            Q:解释最大池化:
            A:最大池化是一种下采样操作，它通过取局部区域中的最大值来减少空间尺寸。这个操作的主要参数有：
                - kernel_size: 池化窗口的大小。
                - stride: 每次移动池化窗口的步幅。
                - padding: 在输入的边缘添加的零填充数量。
            1、kernel_size=2: 每个池化窗口的大小为 2x2。这意味着池化操作会在输入的每个 2x2 区域中取最大值。
            2、stride=2: 每次移动池化窗口时，窗口会跳过两个像素。这意味着池化窗口不会重叠，并且每次都跳过两个像素。
            3、padding=0: 没有填充。池化窗口严格在输入图像的边界内移动。
            """
            self.maxpools.append(nn.MaxPool2d(kernel_size=2,stride=2,padding=0))
        
        # 每个decoder conv前放大一倍图像尺寸，缩小一倍通道数
        self.deconvs=nn.ModuleList()
        for i in range(len(channels)-2):
            self.deconvs.append(nn.ConvTranspose2d(channels[-i-1],channels[-i-2],kernel_size=2,stride=2)) # 不改通道数,尺寸翻倍

        # 每个decoder conv block减少一倍通道数
        self.dec_convs=nn.ModuleList()
        for i in range(len(channels)-2):
            self.dec_convs.append(ConvBlock(channels[-i-1],channels[-i-2],time_emb_size))   # 残差结构

        # 还原通道数,尺寸不变
        self.output=nn.Conv2d(channels[1],img_channel,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x,t):
        # time做embedding
        t_emb=self.time_emb(t)
        
        # encoder阶段
        residual=[]
        for i,conv in enumerate(self.enc_convs):
            x=conv(x,t_emb)
            if i!=len(self.enc_convs)-1:
                residual.append(x)
                x=self.maxpools[i](x)
            
        # decoder阶段
        for i,deconv in enumerate(self.deconvs):
            x=deconv(x)
            residual_x=residual.pop(-1)
            x=self.dec_convs[i](torch.cat((residual_x,x),dim=1),t_emb)    # 残差用于纵深channel维
        return self.output(x) # 还原通道数
        
if __name__=='__main__':

    # 从数据集里面拿数据, 拿batch张图片, [c, h, w] -> [b, c, h, w]
    batch_x=torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(device)
    
    # [0, 1] -> [-1, 1]
    batch_x=batch_x*2-1 
    
    # 随机生成timestep, [0, 1000)
    batch_t=torch.randint(0,timestep,size=(batch_x.size(0),)).to(device)
    
    # 将x0和t传入diffusion的forward中, 利用x_t=....来得到加噪后的图片;
    #  这里的batch_noise_t是加入噪声的多少, 训练的时候用作label。
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t)

    print('batch_x_t:',batch_x_t.size())
    print('batch_noise_t:',batch_noise_t.size())

    # 将加入的噪声以及timestep放入unet(或其他backbone)中进行噪声的预测
    unet=UNet(batch_x_t.size(1)).to(device)
    batch_predict_noise_t=unet(batch_x_t,batch_t)
    
    print('batch_predict_noise_t:',batch_predict_noise_t.size())