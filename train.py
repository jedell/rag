import logging
import torch
from transformers import AdamW
from mistral.model import Transformer as Mistral
from mistral.tokenizer import Tokenizer
from transformers import AutoTokenizer, AutoModel
import loralib as lora
from pathlib import Path
from retriever.index import init_index
from loss import loss_fn
from model import RagModel
from utils import (
    setup_data,
    save_checkpoint,
    load_documents,
    Logger
)


generator_path = "_model"

gtokenizer = Tokenizer(str(Path(generator_path) / "tokenizer.model"))
rtokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)

g = Mistral.from_folder(Path(generator_path))
r = AutoModel.from_pretrained(
    'nomic-ai/nomic-embed-text-v1.5',
    trust_remote_code=True,
    safe_serialization=True,
    rotary_scaling_factor=2
)

matryoshka_dim = 768

ind = init_index(matryoshka_dim, "index/dune.index")
documents_path = "data/chunks"
documents = load_documents(documents_path)

model = RagModel(g, r, gtokenizer, rtokenizer, ind)

# move into RagModel
lora.mark_only_lora_as_trainable(model.generator, bias='all')

batch_size = 32

dataloader, sampler = setup_data(model.retriever_tokenizer, "data/dune.jsonl", batch_size, True)

# init train args
num_epochs = 10
top_k = 5
start_lr = 5e-5
weight_decay = 0.1
steps_per_epoch = len(dataloader)
clip_grad_norm = 1.0
ckpt_freq = 1
log_freq = 1

logger = Logger("dune-rag", {
    "epochs": num_epochs,
    "steps": steps_per_epoch,
    "clip_grad_norm": clip_grad_norm,
    "ckpt_freq": ckpt_freq,
    "log_freq": log_freq,
    "batch_size": batch_size,
    "top_k": top_k,
    "start_lr": start_lr,
    "matryoshka_dim": matryoshka_dim,
})

# Optimizers
optimizer = AdamW(
    model.parameters(),
    lr=start_lr,
    betas=(0.9, 0.95),
    eps=1e-08,
    weight_decay=weight_decay,
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=start_lr,
    total_step=steps_per_epoch,
    pct_start=0.5,
)

def train(epochs_run=0):
    # Training loop  
    for epoch in range(epochs_run, num_epochs):

        sampler.set_epoch(epoch)
        loss = torch.tensor([0.0], device=device)

        for step, batch in enumerate(dataloader):

            optimizer.zero_grad()

            input_ids, labels, mask = batch['input_ids'], batch['labels'], batch['mask']
            retriever_tokens, retriever_attn_mask = batch['retriever_tokens'], batch['retriever_attn_mask']
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            retriever_tokens = retriever_tokens.to(device)
            retriever_attn_mask = retriever_attn_mask.to(device)

            # retrieve
            context_input_ids, context_masks, doc_scores = model.retrieve(batch, documents)
            context_input_ids = context_input_ids.to(device)
            context_masks = context_masks.to(device)

            logits = model.generator(context_input_ids) # context_ids
            # shift
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.transpose(1, 2)

            rag_loss = loss_fn(logits, labels, context_masks, doc_scores)

            rag_loss.backward()
            loss += rag_loss.detach()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        last_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()

        # Logging, validation, saving models, etc.
        if epochs_run % log_freq == 0:
            logger.log({
                "train/loss": loss.item(),
                "train/lr": last_lr,
                "train/steps": step,
                "train/epoch": epoch,
                "train/percent_done": 100 * epochs_run / num_epochs,
                "memory/max_rss": torch.cuda.max_memory_allocated() / 1024**3,
                "memory/allocated": torch.cuda.memory_allocated() / 1024**3,
                "memory/reserved": torch.cuda.memory_reserved() / 1024**3,
            })


        epochs_run += 1

        if (ckpt_freq is not None and epochs_run % ckpt_freq == 0) or epochs_run == num_epochs:
            save_checkpoint(epochs_run, model.generator, model.retriever, optimizer)

