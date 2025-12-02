import torch
import torch.nn as nn
import torchvision.models as models
# 结合 CNN 的局部特征提取能力 + Transformer 的全局依赖建模能力，提高对低分辨率图像中微小病变的分类能力
# 轻量 CNN Backbone：提取局部特征，保持空间信息
#
# Patch Embedding + Transformer Encoder：把 CNN 输出分块 token，捕捉全局依赖
#
# 分类头 (FC)：把 Transformer 输出映射到类别预测
class CNN_Transformer(nn.Module):
    def __init__(self, num_classes=5, img_size=28, patch_size=2, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        # --- CNN Backbone: 轻量级 CNN，保留空间维度 ---
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # [B,64,28,28]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [B,128,14,14]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [B,256,7,7]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # [B,512,7,7]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # --- Patch Embedding ---
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 输出 [B, embed_dim, H_patch, W_patch]

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*2, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Classification Head ---
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # CNN Backbone
        x = self.cnn(x)  # [B,512,H_feat,W_feat]

        # Patch Embedding
        x = self.proj(x)  # [B, embed_dim, H_patch, W_patch]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)  # [B, num_tokens, embed_dim]

        # Transformer Encoder
        x = self.transformer(x)          # [B, num_tokens, embed_dim]
        x = x.mean(dim=1)                # 全局平均池化

        # Classification Head
        x = self.cls_head(x)
        return x

# --- 测试网络 ---
if __name__ == "__main__":
    model = CNN_Transformer(num_classes=5)
    x = torch.randn(32,3,28,28)
    y = model(x)
    print(y.shape)  # [32,5]