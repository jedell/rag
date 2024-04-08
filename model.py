import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from retriever.index import get_top_docs, search
from retriever.nomic import mean_pooling, encode_query, embed
from dataset import IGNORE_INDEX

class RagModel(nn.Module):
    def __init__(
            self,
            generator,
            retriever,
            generator_tokenizer,
            retriever_tokenizer,
            index,
            matryoshka_dim=768,
            top_k=5,
            documents=None
        ):
        super().__init__()
        self.generator = generator
        self.retriever = retriever
        self.generator_tokenizer = generator_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.index = index
        self.matryoshka_dim = matryoshka_dim
        self.top_k = top_k
        self.documents = documents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def process_docs(
        self,
        docs: List[List[str]],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attn_masks: torch.Tensor,
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
                
                assert input_ids[j].shape[0] == labels[j].shape[0] == attn_masks[j].shape[0]
                context_input_ids = torch.cat((doc_tokens, input_ids[j]))
                # ignore index size of doc_tokens cat in front of labels
                context_labels = torch.cat((torch.tensor([IGNORE_INDEX] * len(doc_tokens)), labels[j]))
                context_mask = context_input_ids.ne(self.generator_tokenizer.pad_token_id)


                assert context_input_ids.shape[0] == context_mask.shape[0] == context_labels.shape[0]

                context_inputs['input_ids'].append(context_input_ids)
                context_inputs['masks'].append(context_mask)
                context_inputs['labels'].append(context_labels)
            
        # pad sequences with eos token
        for key in context_inputs:
            if key == 'labels':
                context_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    context_inputs[key],
                    batch_first=True,
                    padding_value=IGNORE_INDEX
                )
            elif key == 'input_ids':
                context_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    context_inputs[key],
                    batch_first=True,
                    padding_value=self.generator_tokenizer.pad_token_id
                )
            elif key == 'masks':
                context_inputs[key] = torch.nn.utils.rnn.pad_sequence(
                    context_inputs[key],
                    batch_first=True,
                    padding_value=False
                )

        return context_inputs['input_ids'], context_inputs['masks'], context_inputs['labels']
    
    def generate(self, prompt):
        encoded_query = encode_query(prompt, self.generator_tokenizer)
        encoded_query = encoded_query.to(self.device)
        embeded_query = embed(encoded_query, self.retriever)
        embeded_query = embeded_query.to(self.device)

        D, I = search(self.index, embeded_query.cpu(), k=1)
        context = [self.documents[i] for i in I.tolist()[0]][0]

        full_prompt = f"<s> [CONTEXT] {context} [/CONTEXT]\n" 
        full_prompt += "[INST]" + prompt + "[/INST]"

        prompt_ids = self.generator_tokenizer(full_prompt, return_tensors="pt")
        prompt_ids = prompt_ids.to(self.device)

        output = self.generator.generate(**prompt_ids, max_length=8192)

        return context, self.generator_tokenizer.decode(output[0], skip_special_tokens=True)

    def retrieve(self, batch, documents):
        input_ids, labels, attn_mask = batch['input_ids'], batch['labels'], batch['attn_mask']
        retriever_inputs, retriever_attn_mask = batch['retriever_tokens'], batch['retriever_attn_mask']
        B = input_ids.shape[0]

        retriever_inputs = retriever_inputs.cuda(non_blocking=True)
        retriever_attn_mask = retriever_attn_mask.cuda(non_blocking=True)

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
        retrieved_doc_embeds = retrieved_doc_embeds.cuda(non_blocking=True)

        # I = (batch_size, top_k), top_k dimension is the document ids
        # assume dataset.get_document(idx) returns tokenized document context ids
        # return context_ids tensor over batched I, context_ids = (batch_size, top_k, max_length)
        docs = []
        for indicies in I:
            docs.append([documents[idx] for idx in indicies])

        context_input_ids, context_masks, context_labels = self.process_docs(docs, input_ids, labels, attn_mask, self.top_k)
        
        # https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/rag/modeling_rag.py#L644
        doc_scores = torch.bmm(
            embeddings_batched.unsqueeze(1),
            retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)

        return context_input_ids, context_masks, context_labels, doc_scores