import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import PatchEmbed
from .time_embedding import TimePositionEmbedding

def adaptive_scale_and_shift(x, scale, shift):
    return x * ( 1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MultiHeadSelfAttention(nn.Module):
    r"""Multi-Head Self-attention Module"""
    def __init__(self, embeding_size, num_heads, qkv_bias = True):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_embedding_size = embeding_size // num_heads
        assert self.head_embedding_size * self.num_heads == embeding_size, \
            "embedding_size should be divisable by num_heads"
        
        # Linear transformations for query, key and value
        self.w_q = nn.Linear(embeding_size, embeding_size, bias = qkv_bias)
        self.w_k = nn.Linear(embeding_size, embeding_size, bias = qkv_bias)
        self.w_v = nn.Linear(embeding_size, embeding_size, bias = qkv_bias)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()

        # Apply Linear transformations
        query = self.w_q(hidden_states)     # [batch_size, seq_len, embedding_size]
        key = self.w_k(hidden_states)       # [batch_size, seq_len, embedding_size]
        value = self.w_v(hidden_states)     # [batch_size, seq_len, embedding_size]

        # Reshape to seperate heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_embedding_size).transpose(1, 2)      # [batch_size, num_heads, seq_len, head_embedding_size]
        key = key.view(batch_size, seq_len, self.num_heads, self.head_embedding_size).transpose(1, 2)          # [batch_size, num_heads, seq_len, head_embedding_size]
        value = value.view(batch_size, seq_len, self.num_heads, self.head_embedding_size).transpose(1, 2)      # [batch_size, num_heads, seq_len, head_embedding_size]

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_embedding_size ** 0.5)
        attention_probs  = F.softmax(attention_scores, dim = -1)

        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value)                # [batch_size, num_heads, seq_len, head_embedding_size]

        # combine heads and return to original size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)    # [batch_size, seq_len, embedding_size]
        return context_layer

class FinalLayer(nn.Module):
    """Final Layer in Diffusion Transformer"""
    def __init__(self, embedding_size, patch_size, out_channels):
        super().__init__()
        self.final_norm = nn.LayerNorm(embedding_size, eps = 1e-6)
        self.linear = nn.Linear(embedding_size, patch_size[0] * patch_size[1] * out_channels, bias = True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, 2 * embedding_size, bias = True)
        )

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim = 1)
        x = adaptive_scale_and_shift(self.final_norm(x), shift, scale)
        x = self.linear(x)
        return x

class DitBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, mlp_ratio = 4.0):
        super(DitBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embedding_size, eps = 1e-6)
        self.attn = MultiHeadSelfAttention(embedding_size, num_heads)
        self.norm2 = nn.LayerNorm(embedding_size, eps = 1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * mlp_ratio, bias = True),
            nn.SiLU(),
            nn.Linear(embedding_size * mlp_ratio, embedding_size, bias = True)
        )
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size * 6, bias = True)
        )

    def forward(self, x, c):
        alpha1, alpha2, beta1, beta2, gamma1, gamma2 = self.adaLN(c).chunk(6, dim = 1)
        x = x + alpha1.unsqueeze(1) * self.attn(adaptive_scale_and_shift(self.norm1(x), gamma1, beta1))
        x = x + alpha2.unsqueeze(1) * self.mlp(adaptive_scale_and_shift(self.norm2(x), gamma2, beta2)) 
        return x
    
class DiT(nn.Module):
    """Diffusion Transformer"""
    def __init__(self, image_size, patch_size, input_channel, embedding_size, num_labels, num_dit_blocks, num_heads, mlp_ratio = 4):
        super(DiT, self).__init__()
        self.out_channels = input_channel
        self.patch_emb = PatchEmbed(patch_size, input_channel, embedding_size)
        self.cls_emb = nn.Embedding(num_embeddings = num_labels, embedding_dim = embedding_size)
        self.time_emb = nn.Sequential(TimePositionEmbedding(embedding_size),
                                      nn.Linear(embedding_size, embedding_size),
                                      nn.GELU(),
                                      nn.Linear(embedding_size, embedding_size))
        

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, f"image_size {image_size} must be divisble by patch size"
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.position_emb = nn.Parameter(torch.zeros(1, self.num_patches, embedding_size), requires_grad = False)

        self.final_layer = FinalLayer(embedding_size, patch_size, input_channel)
        self.dit_blocks = nn.ModuleList()
        for i in range(num_dit_blocks):
            self.dit_blocks.append(DitBlock(embedding_size, num_heads, mlp_ratio))
        self._init_weights()

    def unpatchify(self, x):
        out_channels = self.out_channels
        patch_h, patch_w = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape = (x.shape[0], h, w, patch_h, patch_w, out_channels))
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(shape = (x.shape[0], out_channels, h * patch_h, w * patch_w))
        return imgs

    def _init_weights(self):
        # NOTE: zero init adaLN modulation layers to make output = x
        for blk in self.dit_blocks:
            nn.init.constant_(blk.adaLN[-1].weight, 0)
            nn.init.constant_(blk.adaLN[-1].bias, 0)

        # NOTE: zero init output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, cond):
        x = self.patch_emb(x) + self.position_emb.to(x.device)
        t = self.time_emb(t)
        cond = self.cls_emb(cond)

        # NOTE: add condition and timestep embedding
        cond = t + cond

        for block in self.dit_blocks:
            x = block(x, cond)
        x = self.final_layer(x, cond)
        
        imgs = self.unpatchify(x)
        return imgs

def DiT_XL_2(**kwargs):
    return DiT(embedding_size=1152, num_dit_blocks=28, patch_size=(2, 2), num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(embedding_size=1152, num_dit_blocks=28, patch_size=(4, 4), num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(embedding_size=1152, num_dit_blocks=28, patch_size=(8, 8), num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(embedding_size=1024, num_dit_blocks=24, patch_size=(2, 2), num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(embedding_size=1024, num_dit_blocks=24, patch_size=(4, 4), num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(embedding_size=1024, num_dit_blocks=24, patch_size=(8, 8), num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(embedding_size=768, num_dit_blocks=12, patch_size=(2, 2), num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(embedding_size=768, num_dit_blocks=12, patch_size=(4, 4), num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(embedding_size=768, num_dit_blocks=12, patch_size=(8, 8), num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(embedding_size=384, num_dit_blocks=12, patch_size=(2, 2), num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(embedding_size=384, num_dit_blocks=12, patch_size=(4, 4), num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(embedding_size=384, num_dit_blocks=12, patch_size=(8, 8), num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

if __name__ == "__main__":
    dit_model = DiT_B_2(image_size=(224, 224), input_channel=1, num_labels=10)
    total_params = sum(p.numel() for p in dit_model.parameters())
    print(f"Total number of parameters: {total_params}")
    batch_size = 2  
    input_channels = 1 
    height, width = 224, 224
    
    x = torch.randn(batch_size, input_channels, height, width)
    t = torch.randint(0, 1000, (batch_size,)).float()
    cond = torch.randint(0, 10, (batch_size,))
    
    output = dit_model(x, t, cond)
    
    print(output.shape)