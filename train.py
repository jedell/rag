import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List
from transformers import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from mistral.model import Transformer as Mistral
from mistral.tokenizer import Tokenizer
import transformers
from transformers import AutoTokenizer, AutoModel
import loralib as lora
from pathlib import Path
from retriever.index import init_index, get_top_docs
from retriever.nomic import mean_pooling
from dataset import init_dataset
from utils import setup_data

# Initialize models
matryoshka_dim = 768
generator_path = "_model"

generator_tokenizer = Tokenizer(str(Path(generator_path) / "tokenizer.model"))
retriever_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)

generator = Mistral.from_folder(Path(generator_path))
retriever = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True, rotary_scaling_factor=2)

lora.mark_only_lora_as_trainable(generator, bias='all')

index = init_index(matryoshka_dim, "index/dune.index")

# init train args
num_epochs = 10
top_k = 5

# Optimizers
retriever_optimizer = AdamW(retriever.parameters(), lr=5e-5)
generator_optimizer = AdamW(generator.parameters(), lr=5e-5)

# Scheduler
retriever_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(retriever_optimizer, T_max=num_epochs)
generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=num_epochs)

def marginalize(seq_logits, doc_scores, n_docs=None):
    n_docs = n_docs if n_docs is not None else 1

    # RAG-token marginalization
    seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
        seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
    )
    doc_logprobs = torch.log_softmax(doc_scores, dim=1)
    log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
    return torch.logsumexp(log_prob_sum, dim=1)

def loss_fn(logits, doc_scores, target, reduce_loss=True, epsilon=0.1, n_docs=None):
    
    # shift target left
    target = torch.cat(
        [target[:, 1:], target.new(target.shape[0], 1).fill_(generator_tokenizer.pad_token_id)], 1
    )

    def _mask_pads(ll, smooth_obj):
        pad_mask = target.eq(generator_tokenizer.pad_token_id)
        if pad_mask.any():
            ll.masked_fill_(pad_mask, 0.0)
            smooth_obj.masked_fill_(pad_mask, 0.0)
        return ll.squeeze(-1), smooth_obj.squeeze(-1)

    rag_logprobs = marginalize(logits, doc_scores, n_docs)

    target = target.unsqueeze(-1)
    assert target.dim() == rag_logprobs.dim()

    ll = rag_logprobs.gather(dim=-1, index=target)
    smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits
    ll, smooth_obj = _mask_pads(ll, smooth_obj)
    ll = ll.sum(1)  # sum over tokens
    smooth_obj = smooth_obj.sum(1)

    nll_loss = -ll
    smooth_loss = -smooth_obj

    if reduce_loss:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    eps_i = epsilon / rag_logprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

def process_docs(docs: List[List[str]], input_strings: List[str], n_docs: int):
    context_strings = [
        f"{docs[i][j]}\n{input_strings[i]}"
        for i in range(len(docs))
        for j in range(n_docs)
    ]
    context_inputs = generator_tokenizer.batch_encode_plus(
        context_strings,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=8192
    )
    return context_inputs['input_ids'], context_inputs['attention_mask']

def retrieve(retriever, batch, documents):
    input_ids, retriever_inputs, labels = batch
    B = input_ids.shape[0]

    # embed
    embeded_inputs = retriever(**retriever_inputs)

    embeddings = mean_pooling(embeded_inputs, retriever_inputs['attention_mask'])

    if matryoshka_dim is not None:
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :matryoshka_dim]
    embeddings_batched = F.normalize(embeddings, p=2, dim=1)

    # retrieve()
    I = []
    vectors_batched = []
    for embeddings in embeddings_batched:
        ids, retrieved_doc_embeds = get_top_docs(index, embeddings, top_k)
        I.extend(ids)
        vectors_batched.extend(retrieved_doc_embeds)
    I = np.array(I)
    vectors_batched = np.array(vectors_batched)
    # get embbeddings from index by I

    retrieved_doc_embeds = torch.tensor(vectors_batched)

    # I = (batch_size, top_k), top_k dimension is the document ids
    # assume dataset.get_document(idx) returns tokenized document context ids
    # return context_ids tensor over batched I, context_ids = (batch_size, top_k, max_length)
    docs = [documents[idx] for idx in I]

    input_strings = retriever_tokenizer.batch_decode(retriever_inputs, skip_special_tokens=True)

    context_input_ids, context_attention_mask = process_docs(docs, input_strings, top_k)
    
    # https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/rag/modeling_rag.py#L644
    doc_scores = torch.bmm(
        embeddings_batched.unsqueeze(1),
        retrieved_doc_embeds.transpose(1, 2)
    ).squeeze(1)

    return context_input_ids, context_attention_mask, doc_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader, sampler = setup_data(retriever_tokenizer, "data/dune.jsonl", 32, True)

# config/init vars
epochs_run = 0
steps_per_epoch = len(dataloader)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=num_epochs)

def train():
    # Training loop  
    for epoch in range(epochs_run, num_epochs):

        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):

            generator_optimizer.zero_grad()
            retriever_optimizer.zero_grad()

            input_ids, labels, retriever_inputs = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            retriever_inputs = retriever_inputs.to(device)

            # retrieve
            context_input_ids, context_attention_mask, doc_scores = retrieve(retriever, batch)
            context_input_ids = context_input_ids.to(device)
            context_attention_mask = context_attention_mask.to(device)

            # generate
            logits = generator(input_ids) # context_ids
            # shift
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2)

            # loss
            loss = loss_fn(logits, labels)
            # TODO encorporate retriever loss

            loss.backward()
            retriever_optimizer.step()
            generator_optimizer.step()

            lr_scheduler.step()

        # Logging, validation, saving models, etc.

        epochs_run += 1
        save_checkpoint(epochs_run, generator, retriever, retriever_optimizer, generator_optimizer)

