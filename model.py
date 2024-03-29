import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from retriever.index import get_top_docs
from retriever.nomic import mean_pooling

class RagModel(nn.Module):
    def __init__(
            self,
            generator,
            retriever,
            generator_tokenizer,
            retriever_tokenizer,
            index,
            matryoshka_dim=768,
            top_k=5
        ):
        super().__init__()
        self.generator = generator
        self.retriever = retriever
        self.generator_tokenizer = generator_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.index = index
        self.matryoshka_dim = matryoshka_dim
        self.top_k = top_k

    def process_docs(
        self,
        docs: List[List[str]],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        n_docs: int
    ):
        context_inputs = {
            'input_ids': [],
            'masks': [],
            'labels': []
        }
        for i in range(n_docs):
            for j in range(len(docs)):

                doc_tokens = self.generator_tokenizer.encode(
                    f"<s> [CONTEXT] {docs[j][i]} [/CONTEXT]\n",
                    add_special_tokens=False
                )
                doc_tokens = torch.tensor(doc_tokens)
                doc_mask = torch.tensor([False] * (len(doc_tokens)))

                assert input_ids[i].shape[0] == labels[i].shape[0] == masks[i].shape[0]
                context_tokens = torch.cat((doc_tokens, input_ids[i]))
                context_mask = torch.cat((doc_mask, torch.tensor([False]), masks[i]))

                assert context_tokens.shape[0] == context_mask.shape[0]

                context_inputs['input_ids'].append(context_tokens[:-1])
                context_inputs['masks'].append(context_mask[1:])
                context_inputs['labels'].append(context_tokens[1:])
            
        # pad sequences with eos token
        for key in context_inputs:
            if key == 'labels':
                context_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    context_inputs[key],
                    batch_first=True,
                    padding_value=self.generator_tokenizer.pad_token_id
                )
            else:
                context_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    context_inputs[key],
                    batch_first=True,
                    padding_value=False
                )

        return context_inputs['input_ids'], context_inputs['masks'], context_inputs['labels']

    def retrieve(self, batch, documents):
        input_ids, labels, mask = batch['input_ids'], batch['labels'], batch['mask']
        retriever_inputs, retriever_attn_mask = batch['retriever_tokens'], batch['retriever_attn_mask']
        B = input_ids.shape[0]

        # embed
        embeded_inputs = self.retriever(input_ids=retriever_inputs, attention_mask=retriever_attn_mask)

        embeddings = mean_pooling(embeded_inputs, retriever_attn_mask)

        if self.matryoshka_dim is not None:
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, :self.matryoshka_dim]
        embeddings_batched = F.normalize(embeddings, p=2, dim=1)

        # retrieve()
        I = []
        vectors_batched = []
        for embeddings in embeddings_batched:
            ids, retrieved_doc_embeds = get_top_docs(self.index, embeddings, self.top_k)
            I.extend(ids)
            vectors_batched.extend(retrieved_doc_embeds)
        I = np.array(I)
        vectors_batched = np.array(vectors_batched)
        # get embbeddings from index by I

        retrieved_doc_embeds = torch.tensor(vectors_batched)

        # I = (batch_size, top_k), top_k dimension is the document ids
        # assume dataset.get_document(idx) returns tokenized document context ids
        # return context_ids tensor over batched I, context_ids = (batch_size, top_k, max_length)
        docs = []
        for indicies in I:
            docs.append([documents[idx] for idx in indicies])

        context_input_ids, context_masks, context_labels = self.process_docs(docs, input_ids, labels, mask, self.top_k)
        
        # https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/rag/modeling_rag.py#L644
        doc_scores = torch.bmm(
            embeddings_batched.unsqueeze(1),
            retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)

        return context_input_ids, context_masks, context_labels, doc_scores