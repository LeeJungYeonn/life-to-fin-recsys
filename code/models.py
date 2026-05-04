from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

output_latent_dim = 128


@dataclass
class EncoderOutput:
    hidden: torch.Tensor
    embedding: torch.Tensor
    risk_logits: Optional[torch.Tensor] = None
    ratio_logits: Optional[torch.Tensor] = None
    ratio_probs: Optional[torch.Tensor] = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.hidden
        yield self.embedding


class SourceEncoder(nn.Module):
    def __init__(
        self,
        cardinalities,
        embed_dim=16,
        output_dim=128,
        projection_dim=64,
        num_risk_levels=5,
        ratio_dim=4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.num_risk_levels = num_risk_levels
        self.ratio_dim = ratio_dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, embed_dim) for num_classes in cardinalities]
        )
        total_embed_dim = len(cardinalities) * embed_dim

        self.fc1 = nn.Linear(total_embed_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.proj1 = nn.Linear(output_dim, output_dim)
        self.proj_bn = nn.BatchNorm1d(output_dim)
        self.proj2 = nn.Linear(output_dim, projection_dim)

        hidden_dim = max(output_dim // 2, 32)
        self.risk_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_risk_levels - 1),
        )
        self.ratio_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ratio_dim),
        )

    def forward(self, x_cat):
        embeds = [emb_layer(x_cat[:, idx]) for idx, emb_layer in enumerate(self.embeddings)]
        x_concat = torch.cat(embeds, dim=1)

        hidden = self.fc1(x_concat)
        hidden = self.bn1(hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.bn2(self.fc2(hidden))

        embedding = self.proj1(hidden)
        embedding = self.proj_bn(embedding)
        embedding = F.relu(embedding)
        embedding = self.proj2(embedding)

        risk_logits = self.risk_head(hidden)
        ratio_logits = self.ratio_head(hidden)
        ratio_probs = F.softmax(ratio_logits, dim=1)

        return EncoderOutput(
            hidden=hidden,
            embedding=embedding,
            risk_logits=risk_logits,
            ratio_logits=ratio_logits,
            ratio_probs=ratio_probs,
        )


class TargetEncoder(nn.Module):
    def __init__(
        self,
        input_dim=4,
        output_dim=128,
        projection_dim=64,
        num_risk_levels=5,
        ratio_dim=4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.projection_dim = projection_dim
        self.num_risk_levels = num_risk_levels
        self.ratio_dim = ratio_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.BatchNorm1d(output_dim),
        )

        self.proj1 = nn.Linear(output_dim, output_dim)
        self.proj_bn = nn.BatchNorm1d(output_dim)
        self.proj2 = nn.Linear(output_dim, projection_dim)

        hidden_dim = max(output_dim // 2, 32)
        self.risk_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_risk_levels - 1),
        )
        self.ratio_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ratio_dim),
        )

    def forward(self, x_ratio):
        hidden = self.mlp(x_ratio)

        embedding = self.proj1(hidden)
        embedding = self.proj_bn(embedding)
        embedding = F.relu(embedding)
        embedding = self.proj2(embedding)

        risk_logits = self.risk_head(hidden)
        ratio_logits = self.ratio_head(hidden)
        ratio_probs = F.softmax(ratio_logits, dim=1)

        return EncoderOutput(
            hidden=hidden,
            embedding=embedding,
            risk_logits=risk_logits,
            ratio_logits=ratio_logits,
            ratio_probs=ratio_probs,
        )
