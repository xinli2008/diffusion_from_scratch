import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, image_channel, query_size, value_size, feed_forward_size, cls_embedding_size):
        """
        Initializes the CrossAttention module.

        Args:
            image_channel (int): Number of channels in the input image.
            query_size (int): Dimension of the query vector.
            value_size (int): Dimension of the value vector.
            feed_forward_size (int): Hidden layer size of the feed-forward network.
            cls_embedding_size (int): Dimension of the class embedding vector.
        """
        super(CrossAttention, self).__init__()
        self.wq = nn.Linear(image_channel, query_size)
        self.wk = nn.Linear(cls_embedding_size, query_size)
        self.wv = nn.Linear(cls_embedding_size, value_size)
        self.fc_out = nn.Linear(value_size, image_channel)

        self.norm1 = nn.LayerNorm(image_channel)
        self.feedforward = nn.Sequential(
            nn.Linear(image_channel, feed_forward_size),
            nn.ReLU(),
            nn.Linear(feed_forward_size, image_channel)
        )
        self.norm2 = nn.LayerNorm(image_channel)
    
    def forward(self, input_image, cls_embedding):
        """
        Forward pass of the CrossAttention module.

        Args:
            input_image (torch.Tensor): Input image tensor with shape [batch_size, channel, height, width].
            cls_embedding (torch.Tensor): Class embedding tensor with shape [batch_size, cls_embedding_size].

        Returns:
            attention_out (torch.Tensor): Output tensor after cross-attention processing, with the same shape as the input image.
        """
        hidden_states = input_image          # [b, c, h, w]
        batch_size, channel, height, width = input_image.shape
        hidden_states = hidden_states.reshape([batch_size, height*width, channel])  # [b, h*w, c]
        hidden_states = self.wq(hidden_states)   # [b, h*w, query_size]

        key = self.wk(cls_embedding)      # [b, query_size]
        value = self.wv(cls_embedding)    # [b, value_size]

        key = torch.unsqueeze(key, dim = -1)     # [b, query_size, 1]
        value = torch.unsqueeze(value, dim = 1)                    # [b, 1, value_size]

        # NOTE: [b, h*w, query_size] * [b, quqry_size, 1] = [b, h*w, 1]
        attention_scores = torch.matmul(hidden_states, key) * (hidden_states.shape[-1] ** -0.5)
        attention_scores = F.softmax(attention_scores, dim = -1)

        # NOTE: [b, h*w, 1] * [b, 1, value_size] = [b, h*w, value_size]
        attention_out = torch.matmul(attention_scores, value)

        # NOTE: [b, h*w, value_size] -> [b, h*w, image_channel]
        attention_out = self.fc_out(attention_out)
        attention_out = attention_out.reshape(batch_size, height, width, channel)

        # Optimal:
        x = input_image.permute(0, 2, 3, 1).contiguous()
        attention_out = self.norm1(attention_out + x)
        z = self.feedforward(attention_out)
        attention_out = self.norm2(z + attention_out)

        attention_out = attention_out.permute(0, 3, 1, 2).contiguous()
        
        return attention_out