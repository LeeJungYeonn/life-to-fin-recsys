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


def build_ordinal_pos_weight(labels, num_risk_levels):
    targets = build_ordinal_targets(labels, num_risk_levels)
    positive_counts = targets.sum(dim=0)
    negative_counts = targets.size(0) - positive_counts
    return negative_counts / positive_counts.clamp_min(1.0)


def ordinal_regression_loss(logits, labels, num_risk_levels, pos_weight=None):
    targets = build_ordinal_targets(labels, num_risk_levels)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def ordinal_logits_to_label(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


def ratio_kl_loss(ratio_logits, target_ratio):
    target_ratio = target_ratio.clamp_min(1e-8)
    log_probs = F.log_softmax(ratio_logits, dim=1)
    return F.kl_div(log_probs, target_ratio, reduction="batchmean")


def js_divergence_from_logits(logits, target_probs, eps=1e-8):
    pred_probs = F.softmax(logits, dim=1).clamp_min(eps)
    target_probs = target_probs.clamp_min(eps)
    midpoint = 0.5 * (pred_probs + target_probs)
    pred_kl = torch.sum(pred_probs * (pred_probs.log() - midpoint.log()), dim=1)
    target_kl = torch.sum(target_probs * (target_probs.log() - midpoint.log()), dim=1)
    return 0.5 * (pred_kl + target_kl).mean()


def allocation_l1_loss(logits, target_probs):
    pred_probs = F.softmax(logits, dim=1)
    return torch.mean(torch.abs(pred_probs - target_probs))


def continuous_portfolio_loss(logits, target_probs, js_weight=1.0, l1_weight=1.0):
    loss_js = js_divergence_from_logits(logits, target_probs)
    loss_l1 = allocation_l1_loss(logits, target_probs)
    total = js_weight * loss_js + l1_weight * loss_l1
    return total, {
        "js": float(loss_js.detach().item()),
        "l1": float(loss_l1.detach().item()),
    }


def coral_loss(source_hidden, target_hidden):
    if source_hidden.size(0) < 2 or target_hidden.size(0) < 2:
        return source_hidden.new_tensor(0.0)

    source_centered = source_hidden - source_hidden.mean(dim=0, keepdim=True)
    target_centered = target_hidden - target_hidden.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (source_hidden.size(0) - 1)
    target_cov = target_centered.T @ target_centered / (target_hidden.size(0) - 1)
    return torch.mean((source_cov - target_cov) ** 2)


def pairwise_js_distance(source_probs, target_probs, eps=1e-8):
    source_probs = source_probs.clamp_min(eps)
    target_probs = target_probs.clamp_min(eps)

    source_expand = source_probs.unsqueeze(1)
    target_expand = target_probs.unsqueeze(0)
    midpoint = 0.5 * (source_expand + target_expand)

    source_kl = torch.sum(source_expand * (source_expand.log() - midpoint.log()), dim=2)
    target_kl = torch.sum(target_expand * (target_expand.log() - midpoint.log()), dim=2)
    return 0.5 * (source_kl + target_kl)


def build_cross_modal_positive_mask(
    source_alloc,
    target_alloc,
    labels=None,
    cluster_ids=None,
    js_threshold=0.03,
    include_label_matches=False,
    include_cluster_matches=False,
):
    batch_size = source_alloc.size(0)
    cross_mask = torch.eye(batch_size, dtype=torch.bool, device=source_alloc.device)

    if include_label_matches and labels is not None:
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        cross_mask |= label_mask

    if include_cluster_matches and cluster_ids is not None:
        cluster_mask = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)
        cross_mask |= cluster_mask

    if js_threshold is not None:
        js_distance = pairwise_js_distance(source_alloc, target_alloc)
        cross_mask |= js_distance <= js_threshold

    positive_mask = torch.zeros(
        batch_size * 2,
        batch_size * 2,
        dtype=torch.bool,
        device=source_alloc.device,
    )
    positive_mask[:batch_size, batch_size:] = cross_mask
    positive_mask[batch_size:, :batch_size] = cross_mask.T
    return positive_mask


def multi_positive_supcon_loss(embeddings, positive_mask, temperature=0.15):
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits_mask = ~self_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

    valid_positive_mask = positive_mask & logits_mask
    positive_counts = valid_positive_mask.sum(dim=1)
    valid_anchors = positive_counts > 0
    if not valid_anchors.any():
        return logits.new_tensor(0.0)

    mean_log_prob_pos = (
        (valid_positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1).float()
    )
    return -mean_log_prob_pos[valid_anchors].mean()


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
