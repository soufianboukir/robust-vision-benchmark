import torch
import torch.nn as nn

# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# -----------------------------
# Transformer Block
# -----------------------------
class Block(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# -----------------------------
# ViT-Tiny
# -----------------------------
class ViTTiny(nn.Module):
    def __init__(self, num_classes=10, depth=8, dim=192):
        super().__init__()

        self.patch_embed = PatchEmbed(dim=dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = None  # simplified (can be learned in real ViT)

        self.blocks = nn.Sequential(*[Block(dim) for _ in range(depth)])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])
