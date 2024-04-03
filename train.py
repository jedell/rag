import logging
import fire
import torch
import torch.distributed as dist
from torch.distributed import barrier
import torch.nn.parallel as torch_ddp
import torch.distributed.fsdp.wrap as torch_wrap
from transformers import AdamW
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from retriever.index import init_index
from loss import loss_fn
from model import RagModel
from torch.distributed.fsdp import MixedPrecision
# from finetune.checkpointing import save_checkpoint
from finetune.wrapped_model import build_model, load_initial_model
from finetune.args import TrainArgs
from finetune.utils import TrainState, logged_closing, set_random_seed
from utils import (
    setup_data,
    save_checkpoint,
    load_documents,
    Logger
)
from finetune.distributed import (
    avg_aggregate,
    get_rank,
    get_world_size,
    our_initialize_model_parallel,
    set_device,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator_path = "_model"

batch_size = 1
top_k = 1

args: TrainArgs = TrainArgs.load(Path('config', '7b_lora.yaml'), drop_extra_fields=False)
args.num_microbatches = batch_size * top_k

set_random_seed(args.seed)
set_device()

dist.init_process_group(backend="nccl")

our_initialize_model_parallel("nccl", args.n_replica)

barrier()

matryoshka_dim = 768

ind = init_index(matryoshka_dim, "index/dune.index")
documents_path = "data/chunks"
documents = load_documents(documents_path)

gtokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', model_max_length=8192)
rtokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)
gtokenizer.pad_token = gtokenizer.eos_token

g = build_model(folder=Path('config'), train_args=args)

r = AutoModel.from_pretrained(
    'nomic-ai/nomic-embed-text-v1.5',
    trust_remote_code=True,
    safe_serialization=True,
    rotary_scaling_factor=2
)

model = RagModel(g, r, gtokenizer, rtokenizer, ind)
model.to(device)

ddp_params = {
    "mixed_precision": MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
}

# with torch_wrap.enable_wrap(
#         wrapper_cls=torch_ddp.DistributedDataParallel, **ddp_params
# ):

#     model.retriever = torch_wrap.wrap(model.retriever)

# assert isinstance(model, torch_ddp.DistributedDataParallel)
# logging.info(f"Wrapped model with DDP: {model.retriever}")

with torch_wrap.enable_wrap(
        wrapper_cls=torch_ddp.DistributedDataParallel, **ddp_params
    ):

    # only finetune LoRA parameters and freeze before wrapping
    for name, param in model.generator.named_parameters():
        if "lora" in name or "norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model = torch_wrap.wrap(model)

assert isinstance(model, torch_ddp.DistributedDataParallel)
logging.info(f"Wrapped model with DDP: {model}")

dataloader, sampler, train_ds = setup_data(
    model.module.generator_tokenizer,
    model.module.retriever_tokenizer,
    "data/dune_mistral_instruct.jsonl",
    batch_size,
    True
)
# init train args
num_epochs = 10
start_lr = 5e-5
weight_decay = 0.1
steps_per_epoch = len(dataloader)
clip_grad_norm = 1.0
ckpt_freq = 1
log_freq = 1

logger = Logger(
    "dune-rag",
    {
        "epochs": num_epochs,
        "steps": steps_per_epoch,
        "clip_grad_norm": clip_grad_norm,
        "ckpt_freq": ckpt_freq,
        "log_freq": log_freq,
        "batch_size": batch_size,
        "top_k": top_k,
        "start_lr": start_lr,
        "matryoshka_dim": matryoshka_dim,
    },
    is_master=get_rank() == 0
)

# Optimizers
optimizer = AdamW(
    model.module.parameters(),
    lr=start_lr,
    betas=(0.9, 0.95),
    eps=1e-08,
    weight_decay=weight_decay,
)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=start_lr,
    total_steps=steps_per_epoch,
    pct_start=0.5,
)

load_initial_model(model.module.generator, args.initial_model_path)

model.train()

torch.cuda.empty_cache()

def train(epochs_run=0):
    # Training loop  
    for epoch in range(epochs_run, num_epochs):

        sampler.set_epoch(epoch)
        loss = torch.tensor([0.0], device=device)

        for step, batch in enumerate(dataloader):
            print(step)
            optimizer.zero_grad()

            input_ids, labels, mask = batch['input_ids'], batch['labels'], batch['mask']
            retriever_tokens, retriever_attn_mask = batch['retriever_tokens'], batch['retriever_attn_mask']
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            retriever_tokens = retriever_tokens.to(device)
            retriever_attn_mask = retriever_attn_mask.to(device)

            # retrieve
            context_input_ids, context_masks, context_labels, doc_scores = model.retrieve(batch, documents)

            # decode masked tokens only, from context_masks
            masked_tokens = [token for token, mask in zip(context_labels[0], context_masks[0]) if mask]
            print(model.generator_tokenizer.decode(masked_tokens))
            context_input_ids = context_input_ids.to(device)
            context_masks = context_masks.to(device)

            seqlens = [len(seq) for seq in context_input_ids]
            context_input_ids = context_input_ids.view(-1)


            logits = model.generator.forward(
                input_ids=context_input_ids,
                seqlens=seqlens
            ) # context_ids
            
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

if __name__ == "__main__":

    fire.Fire(train)