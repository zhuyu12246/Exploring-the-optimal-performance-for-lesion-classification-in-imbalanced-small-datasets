import torch
import torch.nn as nn
import torch.nn.functional as F

# Vision Transformer
# 小号 Transformer | 最小尺寸的 Vision Transformer 模型，轻量但性能不错
# Input(3×224×224)
#    │
#    ▼
# Patch Embedding (Conv2d: kernel=16, stride=16)
#    │
#    ▼
# 196 个 patch (每个 patch = 192 维向量)
#    │
#    ▼
# 加上 1 个 CLS token （共 197 tokens）
# 加上 Position Embedding (197×192)
#    │
#    ▼
# Transformer Block × 12 层
#    │
#    ▼
# 取 CLS token (1×192)
#    │
#    ▼
# Linear → 输出 5 类


# ---------------------------------------------------------
# 1. Patch Embedding
# ---------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 14
        self.num_patches = self.grid_size ** 2   # 196

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # B, C, H, W → B, embed_dim, 14, 14 → B, 192, 196 → B, 196, 192
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------
# 2. Multi-Head Self Attention
# ---------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim=192, num_heads=3, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 64
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # qkv: B, N, 3C → 3, B, heads, N, C/heads
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention(Q,K,V)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ---------------------------------------------------------
# 3. MLP
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_features=192, hidden_features=768, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------
# 4. Transformer Block
# ---------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim=192, num_heads=3, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # 残差1
        x = x + self.mlp(self.norm2(x))    # 残差2
        return x


# ---------------------------------------------------------
# 5. Vision Transformer Tiny
# ---------------------------------------------------------
class ViT_Tiny(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 in_chans=3, num_classes=5,
                 embed_dim=192, depth=12, num_heads=3,
                 mlp_ratio=4., drop_rate=0.):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches   # 196

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer Blocks × 12
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, drop_rate, drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)            # (B, 196, 192)

        # Expand cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 192)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, 197, 192)

        # 加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # CLS token as output
        cls = x[:, 0]
        return self.head(cls)


# ---------------------------------------------------------
# 测试模型
# ---------------------------------------------------------
if __name__ == "__main__":
    model = ViT_Tiny(num_classes=5)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("output:", out.shape)
