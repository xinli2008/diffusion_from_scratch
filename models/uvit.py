import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from time_embedding import TimePositionEmbedding
import einops

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = "xformers" 
    except:
        ATTENTION_MODE = "math"
print(f"current attention mode is {ATTENTION_MODE}")

class Attention(nn.Module):
    def __init__(self, hidden_states, num_heads, qkv_bias, qk_scale, attn_drop = 0., proj_drop = 0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        assert hidden_states % num_heads ==0, "num_heads should be divisable by hidden states"
        self.head_dim = hidden_states // num_heads
        self.scale = qk_scale or hidden_states ** -0.5

        self.qkv = nn.Linear(hidden_states, hidden_states * 3, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_states, hidden_states)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x)

        if ATTENTION_MODE == "math":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K = 3, H = self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim = -1)
            attn = self.attn_drop(attn)
            x = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, C)
        elif ATTENTION_MODE == "flash":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K = 3, H = self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)", H = self.num_heads)
        elif ATTENTION_MODE == "xformers":
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            raise ValueError(f"unexcepted ATTENTION_MODE for {ATTENTION_MODE}")
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class UVitBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, act_layer = nn.GELU(),
                 norm_layer = nn.LayerNorm, skip = False, use_checkpoint = False):
        super(UVitBlock, self).__init__()
        self.norm1 = norm_layer(embedding_dim)
        self.attn = Attention(embedding_dim, num_heads, qkv_bias, qk_scale, attn_drop=0., proj_drop=0.)
        self.norm2 = norm_layer(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * mlp_ratio),
            act_layer,
            nn.Linear(embedding_dim * mlp_ratio, embedding_dim)
        )
        self.skip_linear = nn.Linear(2 * embedding_dim, embedding_dim) if skip else None 
    
    def forward(self, x, skip_info = None):
        if self.skip_linear:
            x = self.skip_linear(torch.cat([x, skip_info], dim = -1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class UVit(nn.Module):
    "U-Vit Backbone"
    def __init__(self, img_size = (224, 224), patch_size = (16, 16), in_channels = 3, embedding_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4,
                 qkv_bias = False, qk_scale = None, norm_layer = nn.LayerNorm, num_class = -1, final_conv = True, skip = True):
        super(UVit, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_class = num_class
        self.in_channels = in_channels
        self.patch_size = patch_size

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "patch size shoule be divisable by image size"
        self.patch_emb = PatchEmbed(img_size, patch_size, in_channels, embedding_dim)
        self.time_emb = nn.Sequential(TimePositionEmbedding(embedding_dim),
                                    nn.Linear(embedding_dim, embedding_dim),
                                    nn.GELU(),
                                    nn.Linear(embedding_dim, embedding_dim))
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        if num_class > 0:
            self.label_emb = nn.Embedding(num_class, embedding_dim)
            self.extra_label = 2
        else:
            self.label_emb = nn.Identity()
            self.extra_label = 1
        
        self.position_emb = nn.Parameter(torch.zeros(1, self.extra_label + self.num_patches, embedding_dim))

        self.input_blocks = nn.ModuleList([
            UVitBlock(embedding_dim, num_heads, mlp_ratio, qkv_bias, qk_scale)
            for _ in range(depth // 2)
        ])

        self.mid_block = UVitBlock(embedding_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, skip = False)

        self.out_blocks = nn.ModuleList()
        self.out_blocks = nn.ModuleList([
            UVitBlock(embedding_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, skip=True)
            for _ in range(depth // 2)
        ])

        self.norm = norm_layer(embedding_dim)
        self.patch_dim = patch_size[0] * patch_size[1] * in_channels
        self.decoder_pred = nn.Linear(embedding_dim, self.patch_dim, bias = True)
        self.final_layer = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding=1) if final_conv else nn.Identity()

        trunc_normal_(self.position_emb, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        out_channels = self.in_channels
        patch_h, patch_w = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape = (x.shape[0], h, w, patch_h, patch_w, out_channels))
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        imgs = x.reshape(shape = (x.shape[0], out_channels, h * patch_h, w * patch_w))
        return imgs

    def forward(self, x, t, cond):
        x = self.patch_emb(x)
        B, L, C = x.shape

        time_emb = self.time_emb(t).unsqueeze(dim = 1).to(x.device)
        x = torch.cat((time_emb, x), dim = 1)
        if cond is not None:
            cond_emb = self.label_emb(cond).unsqueeze(dim = 1).to(x.device)
            x = torch.cat((cond_emb, x), dim = 1)
        
        x = x + self.position_emb

        skips = []
        for blk in self.input_blocks:
            x = blk(x)
            skips.append(x)
        
        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())
        
        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extra_label + L
        x = x[:, self.extra_label:, :]
        x = self.unpatchify(x)
        x = self.final_layer(x)
        return x

if __name__ == "__main__":
    uvit_model = UVit(img_size=(224, 224), in_channels=3, num_class=10)
    total_params = sum(p.numel() for p in uvit_model.parameters())
    print(f"Total number of parameters: {total_params}")
    batch_size = 2  
    input_channels = 3 
    height, width = 224, 224
    
    x = torch.randn(batch_size, input_channels, height, width)
    t = torch.randint(0, 1000, (batch_size,)).float()
    cond = torch.randint(0, 10, (batch_size,))
    
    output = uvit_model(x, t, cond)
    
    print(output.shape)