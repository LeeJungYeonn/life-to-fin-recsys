import torch
import torch.nn as nn
import torch.nn.functional as F

output_latent_dim = 128

class SourceEncoder(nn.Module):
    def __init__(self, cardinalities, embed_dim=16, output_dim=128):
        super(SourceEncoder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes in cardinalities
        ])
        total_embed_dim = len(cardinalities) * embed_dim

        # 1. Base Encoder
        self.fc1 = nn.Linear(total_embed_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        # 2. Projection Head (학습용 64차원 변환)
        self.proj1 = nn.Linear(output_dim, output_dim)
        self.proj_bn = nn.BatchNorm1d(output_dim)
        self.proj2 = nn.Linear(output_dim, 64) 

    def forward(self, x_cat):
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            e = emb_layer(x_cat[:, i])
            embeds.append(e)
        x_concat = torch.cat(embeds, dim=1)

        # Base Encoder 통과
        h = self.fc1(x_concat)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.bn2(self.fc2(h)) # h: (추론용) 128차원
        
        # Projection Head 통과
        z = self.proj1(h)
        z = self.proj_bn(z)
        z = F.relu(z)
        z = self.proj2(z)         # z: (학습용) 64차원
        
        # Loss 함수 내부에서 F.normalize를 수행하므로 여기서는 뺐습니다.
        return h, z

class TargetEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super(TargetEncoder, self).__init__()
        # 1. Base Encoder
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim) # 안정성을 위해 BN 추가
        )
        
        # 2. 💡 Target에도 동일한 구조의 Projection Head 추가
        self.proj1 = nn.Linear(output_dim, output_dim)
        self.proj_bn = nn.BatchNorm1d(output_dim)
        self.proj2 = nn.Linear(output_dim, 64)

    def forward(self, x_ratio):
        # Base Encoder 통과
        h_F = self.mlp(x_ratio)     # h_F: (추론용) 128차원
        
        # Projection Head 통과
        z_F = self.proj1(h_F)
        z_F = self.proj_bn(z_F)
        z_F = F.relu(z_F)
        z_F = self.proj2(z_F)       # z_F: (학습용) 64차원
        
        return h_F, z_F