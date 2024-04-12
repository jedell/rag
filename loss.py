import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def marginalize(seq_logits, doc_scores, n_docs=1):
    n_docs = n_docs if n_docs is not None else 1

    # RAG-token marginalization
    seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1)
    doc_logprobs = torch.log_softmax(doc_scores, dim=1)
    doc_logprobs = doc_logprobs.view(-1)
    print(seq_logprobs.shape, doc_logprobs.shape)
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
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, 32000) # TODO move hardcoded value
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)

    # TODO marginalize over doc_scores
    # doc_scores = doc_scores.view(-1, 2)
    # doc_scores = doc_scores.to(shift_logits.device)
    # doc_scores = doc_scores.view(-1)

    # shift_logits = shift_logits + doc_scores.unsqueeze(-1)
    # print(shift_logits.shape, shift_labels.shape)

    cross_entropy_fn = nn.CrossEntropyLoss()

    loss = cross_entropy_fn(shift_logits, shift_labels)

    print(loss)

    return loss