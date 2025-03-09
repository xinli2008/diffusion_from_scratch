import torch
import torch.nn as nn
import math

class TimePositionEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super(TimePositionEmbedding, self).__init__()

        self.embedding_size = embedding_size
        self.half_embedding_size = embedding_size // 2

        half_embedding = torch.exp(
            torch.arange(self.half_embedding_size, dtype=torch.float32) * 
            (-math.log(10000.0) / (self.half_embedding_size - 1))
        )

        # Register the half embedding as a buffer
        self.register_buffer("half_embedding", half_embedding)

    def forward(self, timestep):
        r"""
        Perform TimePositionEmbedding forward
        Args:
            timestep: torch.Tensor, [batch_size]
        Return:
            time_embedding: [batch_size, embedding_size]
        """
        batch_size, device = timestep.shape[0], timestep.device
        timestep = timestep.unsqueeze(1).float() # [batch_size] -> [batch_size, 1]

        half_embedding = self.half_embedding.unsqueeze(0).expand(timestep.shape[0], self.half_embedding_size) # [half_embedding_size] -> [batch_size, half_embedding_size]
        half_embedding_timestep = half_embedding.to(device) * timestep.to(device)  # [batch_size, half_embedding_size] * [batch_size, 1] = [batch_size, half_embedding_size]

        time_embedding = torch.cat([half_embedding_timestep.sin(), half_embedding_timestep.cos()], dim = -1).to(device)
        return time_embedding
    
if __name__ == "__main__":
    timeembedding = TimePositionEmbedding(embedding_size=10)
    timestep = torch.randint(low = 0, high=1000, size = (2,)).to("cuda:0")
    timestep_embedding = timeembedding(timestep)
    print(timestep.shape)
    print(timestep_embedding.shape)