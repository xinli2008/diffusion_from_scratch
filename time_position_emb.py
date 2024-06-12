import torch
import math
from config import *
import torch.nn as nn

class TimePositionEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super(TimePositionEmbedding, self).__init__()
        self.half_embedding_size = embedding_size // 2
        
        # 假如embedding_size =8, 则half_embedding_size = 4(//为向下取整)
        # 则half_embedding = [e**(0*(-1*log(10000)/3)), e**(1*(-1*log(10000)/3)), e**(2*(-1*log(10000)/3)), e**(3*(-1*log(10000)/3))] 
        half_embedding = torch.exp(torch.arange(self.half_embedding_size) * (-1 * math.log(10000) / (self.half_embedding_size -1)))
        
        # 在pytorch中, register_buffer和self.xxx = xxx 都是用来注册信息的。
        # 两者的差异在于:
        # 1、使用register_buffer注册的向量,不会计算梯度,但会包含在模型的state_dict中,用于保存和加载模型状态。
        # 2、使用self.xxx = some_tensor(或者nn.parameter) 定义的向量：会计算梯度，并作为模型参数进行优化。
        self.register_buffer("half_embedding", half_embedding)
    
    def forward(self, timestep):
        r"""
        timestep: [b]
        return: [b, embedding_size]
        """
        timestep = timestep.view(timestep.shape[0], 1)  # [b] -> [b, 1]

        # self.half_embedding: [self.half_embedding_size]
        # unsqueeze -> [1, self.half_embedding_size]
        # expand -> [b, self.half_embedding_size]
        half_embedding = self.half_embedding.unsqueeze(0).expand(timestep.shape[0], self.half_embedding_size)

        # pytorch广播机制, 如何判断两个tensor是否是可以广播的？
        # A: 1、每个tensor至少有一个维度。
        # 2、当对维度大小进行迭代时, 需要从最后一个维度开始向前判断, 维度大小必须相等 or 其中一个维度为1 or 其中一个维度不存在。

        # [b, self.half_embedding_size] * [b, 1] = [b, self.half_embedding_size]
        half_embedding_timestep = half_embedding.to(device) * timestep.to(device)

        # [b, self.half_embedding_size * 2] = [b, embedding_size]
        embedding_timestep = torch.cat((half_embedding_timestep.sin(), half_embedding_timestep.cos()), dim = -1)

        return embedding_timestep

if __name__ == "__main__":
    time_pos_emb = TimePositionEmbedding(embedding_size=10)
    t = torch.randint(low = 0, high = 1000, size = (2,)).to(device)  # [b]
    print(t)
    embedding_t = time_pos_emb(t)
    print(embedding_t)
