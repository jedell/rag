import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from transformers import AdamW
from torch.utils.data import DataLoader
from your_dataset import YourDataset
from mistral.model import Transformer as Mistral
from mistral.tokenizer import Tokenizer
from transformers import AutoTokenizer, AutoModel
import loralib as lora
from pathlib import Path
from retriever.index import init_index, get_top_docs
from retriever.nomic import mean_pooling

# Assuming `YourDataset` is a PyTorch Dataset returning (query, relevant_document, target_text)
dataset = YourDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize models
matryoshka_dim = 768
generator_path = "_model"

generator_tokenizer = Tokenizer(str(Path(generator_path) / "tokenizer.model"))
retriever_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)

generator = Mistral.from_folder(Path(generator_path))
retriever = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True, rotary_scaling_factor=2)

index = init_index(matryoshka_dim, "index/dune.index")

lora.mark_only_lora_as_trainable(generator, bias='all')

# init train args
num_epochs = 10
top_k = 5

# Optimizers
retriever_optimizer = AdamW(retriever.parameters(), lr=5e-5)
generator_optimizer = AdamW(generator.parameters(), lr=5e-5)

# Scheduler
retriever_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(retriever_optimizer, T_max=num_epochs)
generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=num_epochs)

def loss_fn(output, target):
    return -torch.sum(torch.log(torch.softmax(output, dim=1)[range(len(target)), target]))

def process_docs(docs: List[List[str]], input_strings: List[str], n_docs: int):
    context_strings = [
        f"{docs[i][j]}\n{input_strings[i]}"
        for i in range(len(docs))
        for j in range(n_docs)
    ]
    context_inputs = retriever_tokenizer.batch_encode_plus(
        context_strings,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=8192
    )
    return context_inputs['input_ids'], context_inputs['attention_mask']

def train_step(generator, retriever, batch):
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
    docs = [dataset.get_documents(idx) for idx in I]

    input_strings = retriever_tokenizer.batch_decode(docs, skip_special_tokens=True)

    context_input_ids, context_attention_mask = process_docs(docs, input_strings, top_k)
    
    # https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/rag/modeling_rag.py#L644
    doc_scores = torch.bmm(
        embeddings_batched.unsqueeze(1),
        retrieved_doc_embeds.transpose(1, 2)
    ).squeeze(1)

    # generator... TODO


    return loss

# Training loop  
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):

        # Step 1: Encode query with the retriever
        input_ids = tokenizer(query, return_tensors="pt", padding=True, truncation=True).input_ids
        query_embeddings = retriever(input_ids).last_hidden_state.mean(dim=1)  # Simplified

        # Step 2: Retrieve documents based on query embeddings
        # This step is highly dependent on your retrieval mechanism.
        # For simplicity, let's assume `relevant_document` is what we retrieve.

        # Step 3: Prepare input for the generator
        # This might involve concatenating the query and relevant document, encoding them, etc.
        generator_input = tokenizer(relevant_document, return_tensors="pt", padding=True, truncation=True).input_ids

        # Step 4: Generate output with Mistral
        generator_output = generator(generator_input)

        # Step 5: Calculate loss (assuming a simple case where both models are trained to minimize the same loss)
        # This is a placeholder; your actual loss calculation will depend on your models and task.
        loss = compute_loss(generator_output, target_text)

        # Step 6: Backpropagation
        retriever_optimizer.zero_grad()
        generator_optimizer.zero_grad()
        loss.backward()
        retriever_optimizer.step()
        generator_optimizer.step()

        # Logging, validation, saving models, etc.