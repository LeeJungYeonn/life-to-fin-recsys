import torch
import torch.nn as nn

output_latent_dim = 128

class SourceEncoder(nn.Module):
    def __init__(self, cardinalities, embed_dim=16, output_dim=128):
        super(SourceEncoder, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes in cardinalities
        ])
        total_embed_dim = len(cardinalities) * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x_cat):
        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            e = emb_layer(x_cat[:, i])
            embeds.append(e)
        x_concat = torch.cat(embeds, dim=1)
        z_NF = self.mlp(x_concat)
        return z_NF

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