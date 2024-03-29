import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def marginalize(seq_logits, doc_scores, n_docs=None):
    n_docs = n_docs if n_docs is not None else 1

    # RAG-token marginalization
    seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
        seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
    )
    doc_logprobs = torch.log_softmax(doc_scores, dim=1)
    log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
    return torch.logsumexp(log_prob_sum, dim=1)

def cross_entropy(logits: torch.Tensor, target: torch.Tensor, reduction: str):
    assert reduction in ["mean", "none"]
    mb_loss = F.cross_entropy(logits, target, reduction=reduction)
    return mb_loss

def compute_loss_with_mask(
    logits: torch.Tensor, target: torch.Tensor, target_mask: Optional[torch.Tensor]
):
    if target_mask is not None:
        mb_loss = cross_entropy(logits, target, reduction="none")
        mb_loss = torch.sum(mb_loss * target_mask) / torch.sum(target_mask)
    else:
        mb_loss = cross_entropy(logits, target, reduction="mean")
    return mb_loss

def loss_fn(logits, target, target_mask, doc_scores, reduce_loss=True, epsilon=0.1, n_docs=None):
    
    # ce_loss = compute_loss_with_mask(logits, target, target_mask)

    def _mask_pads(ll, smooth_obj, mask):
        mask = ~mask
        if mask.any():
            ll.masked_fill_(mask, 0.0)
            smooth_obj.masked_fill_(mask, 0.0)
        return ll.squeeze(-1), smooth_obj.squeeze(-1)

    rag_logprobs = marginalize(logits, doc_scores, n_docs)

    target = target.unsqueeze(-1)
    assert target.dim() == rag_logprobs.dim()

    ll = rag_logprobs.gather(dim=-1, index=target)
    smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
    ll, smooth_obj = _mask_pads(ll, smooth_obj, target_mask)
    ll = ll.sum(1)  # sum over tokens
    smooth_obj = smooth_obj.sum(1)

    nll_loss = -ll
    smooth_loss = -smooth_obj

    if reduce_loss:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / rag_logprobs.size(-1)
    rag_loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    loss = rag_loss # + ce_loss
    return loss