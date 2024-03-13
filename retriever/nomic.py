import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

sentences = ['search_query: What is TSNE?', 'search_query: Who is Laurens van der Maaten?']

def init_retriever(seq_len=8192):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=seq_len)
    retriever = AutoModel.from_pretrained(
        'nomic-ai/nomic-embed-text-v1.5', 
        trust_remote_code=True, 
        safe_serialization=True,
        rotary_scaling_factor=2
        )

    return retriever, tokenizer

def encode_document(document: str, tokenizer: AutoTokenizer):
    """
    Tokenize document for retriever
    """
    if not document.startswith('search_document:'):
        document = f'search_document: {document}'
    encoded_document = tokenizer(document, padding=True, truncation=True, return_tensors="pt")

    return encoded_document

def encode_query(query: str, tokenizer: AutoTokenizer):
    """
    Tokenize query for retriever
    """
    if not query.startswith('search_query:'):
        query = f'search_query: {query}'
    encoded_query = tokenizer(query, padding=True, truncation=True, return_tensors="pt")

    return encoded_query

def embed(inputs, retriever, matryoshka_dim=None, eval=True):
    """
    Embed text with retriever
    """
    if not eval:
        retriever.train()
    else:
        retriever.eval()

    if eval:
        with torch.no_grad():
            embeded_inputs = retriever(**inputs)
    else:
        embeded_inputs = retriever(**inputs)

    embeddings = mean_pooling(embeded_inputs, inputs['attention_mask'])

    if matryoshka_dim is not None:
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :matryoshka_dim]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

    

    




