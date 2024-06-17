import torch
import torch.nn as nn
from config import *
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, channel, query_size, value_size, cls_embedding_size):
        super(CrossAttention, self).__init__()
        self.w_q = nn.Linear(channel, query_size)
        self.w_k = nn.Linear(cls_embedding_size, query_size)
        self.w_v = nn.Linear(cls_embedding_size, value_size)
        self.fc_out = nn.Linear(value_size, channel)
        self.norm = nn.BatchNorm2d(channel)

    def forward(self, x, cls_embedding):
        r"""
        Perform CrossAttention forward process
        Args:
            x: tensor, [b, c, w, h]
            cls_embedding: tensor, [b, cls_embedding_size]
        Return:
            tensor, attention output
        """        
        hidden_states = x.permute(0,1,3,2)  

        batch_size, channel, width, height = hidden_states.shape
        hidden_states = hidden_states.reshape([batch_size, width*height, channel]) 

        query = self.w_q(hidden_states)  # [b, width*height, query_size]
        key = self.w_k(cls_embedding) # [b, cls_embedding_size] -> [b, query_size]
        value = self.w_v(cls_embedding) # [b, cls_embedding_size] -> [b, value_size]

        key = torch.unsqueeze(key, dim = 2)  # [b, query_size] -> [b, query_size, 1]
        value = torch.unsqueeze(value, dim = 1)  # [b, value_size] -> [b, 1, value_size]

        # attention calculation
        attention_scores = torch.matmul(query, key) * (query.shape[-1] ** -0.5)  # [b, width*height, 1]
        attention_scores = F.softmax(attention_scores, dim = -1) # [b, width*height, 1]

        attention_out = torch.matmul(attention_scores, value)  # [b, width*height, value_size]
        attention_out = self.fc_out(attention_out)  # [n, width*height, channel]
        attention_out = attention_out.reshape(batch_size, width, height, channel)  
        attention_out = attention_out.permute(0, 3, 1, 2)  # [b, c, w, h]
        attention_out = self.norm(attention_out)

        return attention_out

if __name__ == "__main__":
    batch_size, channel, query_size, value_size, cls_embedding_size = 2, 1, 256, 128, 32
    cross_attn = CrossAttention(channel = channel, query_size = query_size, value_size = value_size, cls_embedding_size = cls_embedding_size)

    input_tensor = torch.randn([batch_size, channel, image_size, image_size])
    cls_embedding = torch.randn([batch_size, cls_embedding_size])

    output_tensor = cross_attn(input_tensor, cls_embedding)
    print(output_tensor.shape)
