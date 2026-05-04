import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_infonce_loss(source_embedding, target_embedding, temperature=0.1):
    source_embedding = F.normalize(source_embedding, dim=1)
    target_embedding = F.normalize(target_embedding, dim=1)

    logits = source_embedding @ target_embedding.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_source = F.cross_entropy(logits, labels)
    loss_target = F.cross_entropy(logits.T, labels)
    return (loss_source + loss_target) / 2.0


def build_ordinal_targets(labels, num_risk_levels):
    thresholds = torch.arange(num_risk_levels - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds).float()


def ordinal_regression_loss(logits, labels, num_risk_levels):
    targets = build_ordinal_targets(labels, num_risk_levels)
    return F.binary_cross_entropy_with_logits(logits, targets)


def ordinal_logits_to_label(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


def ratio_kl_loss(ratio_logits, target_ratio):
    target_ratio = target_ratio.clamp_min(1e-8)
    log_probs = F.log_softmax(ratio_logits, dim=1)
    return F.kl_div(log_probs, target_ratio, reduction="batchmean")


def batch_centroid_alignment_loss(source_embedding, target_embedding, labels, num_risk_levels):
    source_embedding = F.normalize(source_embedding, dim=1)
    target_embedding = F.normalize(target_embedding, dim=1)

    losses = []
    for cls_idx in range(num_risk_levels):
        mask = labels == cls_idx
        if mask.any():
            source_centroid = source_embedding[mask].mean(dim=0)
            target_centroid = target_embedding[mask].mean(dim=0)
            losses.append(F.mse_loss(source_centroid, target_centroid))

    if not losses:
        return source_embedding.new_tensor(0.0)
    return torch.stack(losses).mean()


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, lambda_weight):
        ctx.lambda_weight = lambda_weight
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_weight * grad_output, None


def grad_reverse(tensor, lambda_weight=1.0):
    return _GradientReversal.apply(tensor, lambda_weight)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, embedding, lambda_weight=1.0):
        return self.net(grad_reverse(embedding, lambda_weight=lambda_weight))


def domain_confusion_loss(domain_discriminator, source_embedding, target_embedding, lambda_weight=1.0):
    source_embedding = F.normalize(source_embedding, dim=1)
    target_embedding = F.normalize(target_embedding, dim=1)

    embeddings = torch.cat([source_embedding, target_embedding], dim=0)
    domain_labels = torch.cat(
        [
            torch.zeros(source_embedding.size(0), dtype=torch.long, device=source_embedding.device),
            torch.ones(target_embedding.size(0), dtype=torch.long, device=target_embedding.device),
        ],
        dim=0,
    )

    logits = domain_discriminator(embeddings, lambda_weight=lambda_weight)
    loss = F.cross_entropy(logits, domain_labels)
    accuracy = (logits.argmax(dim=1) == domain_labels).float().mean()
    return loss, accuracy
