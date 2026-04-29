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

        # 1. Base Encoder: 실제 데이터의 핵심 특징(Representation)을 뽑는 역할
        self.fc1 = nn.Linear(total_embed_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)           # 원본에 있던 Dropout 유지
        self.fc2 = nn.Linear(256, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        # 2. Projection Head: Contrastive 학습(Loss 계산)을 위해 차원을 변환하는 역할 (학습 끝나면 버림)
        self.proj1 = nn.Linear(output_dim, output_dim)
        self.proj_bn = nn.BatchNorm1d(output_dim)
        self.proj2 = nn.Linear(output_dim, 64)   # 차원을 64로 줄임 (Bottleneck)


    def forward(self, x_cat):
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            e = emb_layer(x_cat[:, i])
            embeds.append(e)
        x_concat = torch.cat(embeds, dim=1)

        # 1. Base Encoder 통과
        h = self.fc1(x_concat)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.bn2(self.fc2(h)) # h: (추론용) Representation (크기: output_dim)
        
        # 2. Projection Head 통과
        z = self.proj1(h)
        z = self.proj_bn(z)
        z = F.relu(z)
        z = self.proj2(z)         # z: (학습용) Contrastive Space (크기: 64)
        
        # 3. L2 정규화 (모델 단에서 적용)
        z = F.normalize(z, dim=1) 
        
        # h와 z 두 개를 모두 반환합니다!
        return h, z

class TargetEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=128):
        super(TargetEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x_ratio):
        z_F = self.mlp(x_ratio)
        return z_F