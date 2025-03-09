import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_attention import CrossAttention
from .time_embedding import TimePositionEmbedding

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_embedding_size, query_size, value_size, feed_forward_size, cls_embedding_size):
        r"""
        Initializes the ConvBlock module.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            time_embedding_size (int): Dimension of the time embedding vector.
            query_size (int): Dimension of the query vector for cross-attention.
            value_size (int): Dimension of the value vector for cross-attention.
            feed_forward_size (int): Hidden layer size of the feed-forward network in cross-attention.
            cls_embedding_size (int): Dimension of the class embedding vector.
        """
        super(ConvBlock, self).__init__()

        self.sequence1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.time_embedding_linear = nn.Linear(time_embedding_size, out_channel)
        self.relu = nn.ReLU()

        self.sequence2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.cross_attention = CrossAttention(image_channel = out_channel, query_size = query_size, value_size = value_size, 
                                              feed_forward_size = feed_forward_size, cls_embedding_size = cls_embedding_size)

    
    def forward(self, hidden_states, time_embedding, cls_embedding):
        r"""
        Forward pass of the ConvBlock module.

        Args:
            hidden_states (torch.Tensor): Input tensor with shape [batch_size, in_channel, height, width].
            time_embedding (torch.Tensor): Time embedding tensor with shape [batch_size, time_embedding_size].
            cls_embedding (torch.Tensor): Class embedding tensor with shape [batch_size, cls_embedding_size].

        Returns:
            hidden_states (torch.Tensor): Output tensor with shape [batch_size, out_channel, height, width].
        """
        hidden_states = self.sequence1(hidden_states)  # [b, output_channel, h, w]
        # [b, time_embedding_size] -> [b, out_channel] -> [b, out_channel, 1, 1]
        time_embedding = self.relu(self.time_embedding_linear(time_embedding).view(hidden_states.shape[0], hidden_states.shape[1], 1, 1))
        
        # NOTE：fuse the image features and time embedding
        hidden_states = hidden_states + time_embedding
        
        # NOTE: fuse the image features and cls embedding
        hidden_states = self.cross_attention(hidden_states, cls_embedding)

        return hidden_states
    
class UnetModel(nn.Module):
    def __init__(self, input_channel, channels = [64, 128, 256, 512, 1024], time_embedding_size = 256, query_size = 16, value_size = 16, 
                 feed_forward_size = 32, cls_embedding_size = 32):
        super(UnetModel, self).__init__()

        channels = [input_channel] + channels

        # timestep -> time embedding
        self.time_embedding = nn.Sequential(
            TimePositionEmbedding(time_embedding_size),
            nn.Linear(time_embedding_size, time_embedding_size),
            nn.ReLU()
        )

        # classification label -> cls embedding
        self.cls_embedding = nn.Embedding(10, cls_embedding_size)

        self.unet_encoder = nn.ModuleList()
        for i in range(len(channels) -1):
            self.unet_encoder.append(ConvBlock(channels[i], channels[i+1], time_embedding_size, query_size, value_size, feed_forward_size, cls_embedding_size))
        
        self.maxpools = nn.ModuleList()
        for i in range(len(channels) -2):
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
        
        self.dwconv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.dwconv.append(nn.ConvTranspose2d(channels[-i-1],channels[-i-2], kernel_size = 2, stride = 2))

        self.unet_decoder=nn.ModuleList()
        for i in range(len(channels)-2):
            self.unet_decoder.append(ConvBlock(channels[-i-1],channels[-i-2],time_embedding_size, query_size, value_size, feed_forward_size, cls_embedding_size))

        # NOTE: 1x1 conv 
        self.output_conv = nn.Conv2d(channels[1], input_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, input_image, input_timestep, input_classification_label):
        
        input_time_embedding = self.time_embedding(input_timestep)
        input_cls_embedding = self.cls_embedding(input_classification_label)

        # NOTE: unet encoder
        residual = []
        for i, conv in enumerate(self.unet_encoder):
            x = conv(input_image, input_time_embedding, input_cls_embedding)
            if i!= len(self.unet_encoder) -1 :
                residual.append(x)
                input_image = self.maxpools[i](x)
        
        # NOTE: unet decoder
        for i, deconv in enumerate(self.dwconv):
            x = deconv(x)
            residual_info = residual.pop(-1)
            x = self.unet_decoder[i](torch.cat([residual_info, x], dim = 1), input_time_embedding, input_cls_embedding)

        return self.output_conv(x)