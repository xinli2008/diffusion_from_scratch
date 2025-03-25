import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, input_channels, hidden_states):
        super(PatchEmbed, self).__init__()
        self.projection = nn.Conv2d(input_channels, hidden_states, patch_size, patch_size)
        

    def forward(self, x):
        hidden_states = self.projection(x)    # [b, hidden_states, patch_size, patch_size]
        hidden_states = hidden_states.flatten(2).transpose(-1, -2)
        return hidden_states
