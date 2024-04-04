import os
import logging
from typing import List

import torch
import loralib as lora

import torch
from typing import List
from torch.utils.data import DataLoader, DistributedSampler
import transformers
from transformers import AutoTokenizer
import loralib as lora
from dataset import init_dataset, build_mistral_instruct_dataset
from finetune.args import TrainArgs
from finetune.distributed import get_rank, get_world_size
from finetune.wrapped_model import build_model, PARALLEL_MODEL

logger = logging.getLogger(__name__)

def setup_data(
    generator_tokenizer: transformers.PreTrainedTokenizer,
    retriever_tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = False,
    num_workers: int = 0,
):
    # dm = init_dataset(generator_tokenizer, retriever_tokenizer, data_path)
    dm = build_mistral_instruct_dataset(data_path, generator_tokenizer, retriever_tokenizer)

    sampler = DistributedSampler(
        dataset=dm['train'],
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=0,
        drop_last=True,
    )
    dataloader = DataLoader(
        dm['train'],
        batch_size=batch_size,
        collate_fn=dm['collator'],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataloader, dm['train']

def setup_model(args: TrainArgs):

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

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Rank {get_rank():.0f} has {train_params:,.0f} params to finetune")

    return model

@torch.no_grad()
def save_checkpoint(
        epoch: int,
        total_epochs: int,
        max_steps_per_epoch: int,
        optimizer: torch.optim.Optimizer,
        model: PARALLEL_MODEL,
    ) -> None:
    """
    Checkpoint the state of the recipe. The constructed checkpoint state dict
    contains the following information:
    - Merged weights with key MODEL_KEY
    - Adapter weights with key ADAPTER_KEY
    - Relevant recipe state if training is not complete

    Checkpointer will save the merged weights, adapter weights and recipe state in
    different checkpoint files. To correctly resume from training, the adapter weights
    and recipe state must be provided along with the base model weights.
    """
    ckpt_dict = {}
    # if training is in-progress, checkpoint the optimizer state as well
    if epoch + 1 < total_epochs:
        ckpt_dict.update(
            {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "total_epochs": total_epochs,
                "max_steps_per_epoch": max_steps_per_epoch,
            }
        )

    this_model = model.module

    # construct state dict with only lora weights
    lora_state_dict = lora.lora_state_dict(generator)

    ckpt_dict.update({"generator": lora_state_dict})
    
    # save retriever
    retriever_state_dict = {k: v.cpu() for k, v in retriever.state_dict().items()}
    ckpt_dict.update({"retriever": retriever_state_dict})


def load_documents(path: str, chunk_size: int = None) -> List[str]:
    """
    Load documents from a specified path and optionally chunk the text based on a specified chunk size.
    This can handle both a single document or a directory of documents.
    Each document is read as a string, optionally chunked, and returned in a list.
    """
    documents = []
    if os.path.isdir(path):
        # If path is a directory, recursively iterate over all files in the directory and subdirectories
        for root, dirs, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        if chunk_size is not None:
                            print(f"Chunking {file_path}, chunk size: {chunk_size}")
                            documents.extend(chunk_text(content, chunk_size))
                        else:
                            documents.append(content)
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                        if chunk_size is not None:
                            print(f"Chunking {file_path}, chunk size: {chunk_size}")
                            documents.extend(chunk_text(content, chunk_size))
                        else:
                            documents.append(content)
    elif os.path.isfile(path):
        # If path is a single file, read the file
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                if chunk_size is not None:
                    print(f"Chunking {path}, chunk size: {chunk_size}")
                    documents.extend(chunk_text(content, chunk_size))
                else:
                    documents.append(content)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as file:
                content = file.read()
                if chunk_size is not None:
                    print(f"Chunking {path}, chunk size: {chunk_size}")
                    documents.extend(chunk_text(content, chunk_size))
                else:
                    documents.append(content)
    else:
        raise ValueError(f"Path {path} is neither a valid file nor a directory.")
    
    return documents

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Divide a text into semantic pieces of a specified size. This is a naive implementation that simply chunks
    the text by a character count. More sophisticated methods could be used for semantic chunking.
    i.e https://python.langchain.com/docs/modules/data_connection/document_transformers/
    """
    # return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # docs = text_splitter.create_documents([text])
    # print("Chunks:", len(docs))
    # return [doc.page_content for doc in docs]
    return text


import wandb

class Logger:
    def __init__(self, project_name, config=None, is_master=True):
        self.project_name = project_name
        self.config = config
        self.is_master = is_master
        if self.is_master:
            wandb.init(project=self.project_name, config=self.config)
    
    def log(self, metrics, step=None):
        if self.is_master:
            wandb.log(metrics, step=step)

if __name__ == "__main__":
    # documents = load_documents("data/Dune 1 Dune.txt", chunk_size='semantic')
    
    # # save chunks to separate files
    # os.makedirs("data/chunks/Dune 1 Dune", exist_ok=True)
    # for i, doc in enumerate(documents):
    #     with open(f"data/chunks/Dune 1 Dune/dune1_chunk_{i}.txt", "w") as f:
    #         f.write(doc)

    from transformers import AutoTokenizer

    retriever_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)

    texts = [
        "Dune 1 Dune.txt",
        "Dune 2 Dune Messiah.txt",
        "Dune 3 Children of Dune.txt",
        "Dune 4 God Emperor.txt",
        "Dune 5 Heretics of Dune.txt",
        "Dune 6 Chapterhouse.txt",
    ]
    texts = [f"data/{text}" for text in texts]

    for out_idx, text_path in enumerate(texts):
        # split Dune 1 by '===' indicating chapters and further split chapters if needed
        with open(text_path, "r") as f:
            dune_text = f.read()
        chapters = dune_text.split("===")
        print(f"Processing {text_path}, len(chapters): {len(chapters)}")
        
        sub_chapters = []
        for idx, chapter in enumerate(chapters):
            tokens = retriever_tokenizer(chapter, return_tensors='pt')
            token_len = tokens['input_ids'].shape[1]
            
            if token_len > 2048:
                num_splits = 2 
                while True:
                    sub_chapter_len = len(chapter) // num_splits
                    sub_chapters_split = []
                    sub_chapter_char_lens = []
                    start = 0
                    while start < len(chapter):
                        end = start + sub_chapter_len
                        if end < len(chapter):

                            while end < len(chapter) and chapter[end] != '\n':
                                end += 1

                        sub_chapters_split.append(chapter[start:end])
                        sub_chapter_char_lens.append((start, end))
                        start = end
                        
                    sub_chapter_tokens = [retriever_tokenizer(sub_chapter, return_tensors='pt')['input_ids'].shape[1] for sub_chapter in sub_chapters_split]
                    if all(sub_len <= 2048 for sub_len in sub_chapter_tokens):
                        sub_chapter_texts = [chapter[start:end] for start, end in sub_chapter_char_lens]
                        if len(sub_chapter_texts) > num_splits:
                            sub_chapter_texts[-2] += sub_chapter_texts[-1]
                            sub_chapter_texts.pop(-1)
                        sub_chapters.extend(sub_chapter_texts)
                        break
                    else:
                        num_splits += 1
            else:
                sub_chapters.append(chapter)
        
        print(f"Total sub-chapters: {len(sub_chapters)}")

        # save sub-chapters to separate files
        os.makedirs(f"data/chunks/dune{out_idx+1}", exist_ok=True)
        # remove all from data/chunks/dune{out_idx+1}
        for file in os.listdir(f"data/chunks/dune{out_idx+1}"):
            os.remove(f"data/chunks/dune{out_idx+1}/{file}")

        for i, sub_chapter in enumerate(sub_chapters):
            with open(f"data/chunks/dune{out_idx+1}/dune{out_idx+1}_sub_chapter_{i+1}.txt", "w") as f:
                f.write(sub_chapter.strip().replace("\t", "").replace("    ", "").replace("\n\n", "\n"))


    

