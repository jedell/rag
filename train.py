import logging
import fire
import torch
import torch.distributed as dist
from pathlib import Path
from retriever.index import init_index
from loss import loss_fn
from finetune.checkpointing import save_checkpoint
from finetune.args import TrainArgs
from finetune.utils import set_random_seed
from utils import (
    setup_data,
    setup_model,
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

log = logging.getLogger(__name__)

def train():
    batch_size = 2
    top_k = 2

    args: TrainArgs = TrainArgs.load(Path('config', '7b_lora.yaml'), drop_extra_fields=False)
    args.num_microbatches = batch_size * top_k

    set_random_seed(args.seed)
    set_device()

    dist.init_process_group(backend="nccl")

    our_initialize_model_parallel("nccl", args.n_replica)

    matryoshka_dim = 768

    index = init_index(matryoshka_dim, "index/dune.index")
    documents_path = "data/chunks"
    documents = load_documents(documents_path)

    model = setup_model(index, documents, training=True, quantization=True, top_k=top_k)

    print(f"Rank: {get_rank()}, World size: {get_world_size()}")

    dataloader, train_ds = setup_data(
        model.module.generator_tokenizer,
        model.module.retriever_tokenizer,
        "data/dune_mistral_instruct.jsonl",
        batch_size,
        False,
        rank=get_rank(),
        world_size=get_world_size()
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=start_lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=start_lr,
        total_steps=steps_per_epoch,
        pct_start=0.5,
    )

    def get_all_reduce_mean(tensor):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor = tensor / torch.distributed.get_world_size()
        return tensor

    model.train()
    dist.barrier()

    torch.cuda.empty_cache()

    epochs_run = 0
    local_rank = get_rank()
    # Training loop  
    for epoch in range(epochs_run, num_epochs):

        dataloader.sampler.set_epoch(epoch)
        loss = torch.tensor([0.0], device='cuda')

        for step, batch in enumerate(dataloader):
            is_last_step = step == len(dataloader) - 1

            input_ids, labels, attn_mask = batch['input_ids'], batch['labels'], batch['attn_mask']
            retriever_tokens, retriever_attn_mask = batch['retriever_tokens'], batch['retriever_attn_mask']

            # retrieve
            context_input_ids, context_masks, context_labels, doc_scores = model.module.retrieve(batch, documents)
            
            context_input_ids = context_input_ids.cuda(non_blocking=True)
            context_masks = context_masks.cuda(non_blocking=True)
            context_labels = context_labels.cuda(non_blocking=True)

            out = model.module.generator(
                input_ids=context_input_ids,
                labels=context_labels
            )
            
            logits = out.logits

            # shift
            # logits = logits[..., :-1, :].contiguous()
            # labels = labels[..., 1:].contiguous()
            # logits = logits.transpose(1, 2)

            rag_loss = loss_fn(logits, context_labels, context_masks, doc_scores, n_docs=top_k)

            rag_loss.backward()

            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()
            last_lr = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            loss = rag_loss.detach()

            avg_loss = get_all_reduce_mean(loss.item()) 

            if not is_last_step:
                assert args.num_microbatches > 1
                torch.cuda.synchronize()

            if args.num_microbatches > 1:
                loss /= args.num_microbatches
                for p in model.parameters():
                    if p.requires_grad:
                        assert p.grad is not None
                        p.grad.div_(args.num_microbatches)

            # Logging, validation, saving models, etc.
            if step % log_freq == 0:
                if local_rank == 0:
                    logger.log({
                        "train/avg_loss": avg_loss,
                        "train/lr": last_lr,
                        "train/steps": step,
                        "train/epoch": epoch,
                        "train/percent_done": 100 * epochs_run / num_epochs,
                        "memory/max_rss": torch.cuda.max_memory_allocated() / 1024**3,
                        "memory/allocated": torch.cuda.memory_allocated() / 1024**3,
                        "memory/reserved": torch.cuda.memory_reserved() / 1024**3,
                    })
                # log epcoch,percent done, loss
                log.info(
                    f"Epoch {epoch}. Percent done: {100 * step / len(dataloader):.2f}. Loss: {avg_loss:.4f}"
                )


                if (ckpt_freq is not None and step % ckpt_freq == 0) or step == len(dataloader) - 1:
                    save_checkpoint(model, epochs_run, args.run_dir)

        epochs_run += 1

if __name__ == "__main__":
    fire.Fire(train)