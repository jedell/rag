import os
import numpy as np
import faiss
from typing import Tuple
from faiss import write_index, read_index
from .nomic import embed, encode_document

# TODO: look at FAISS impl here
# https://www.llamaindex.ai/open-source
# https://python.langchain.com/docs/modules/data_connection/vectorstores/

def init_index(embed_dim, index_path=None, M=16, aux_dim=False):
    print("Initializing index...")
    if index_path and os.path.exists(index_path):
        return read_index(index_path)
    # we can experiment with aux dimension later...
    if aux_dim:
        embed_dim += 1
    index = faiss.IndexHNSWFlat(embed_dim, M)
    print(f"Index initialized, dim: {embed_dim}, M: {M}")
    return index

def search(index, query, k, aux_dim=False):
    if aux_dim:
        aux_dim = np.zeros(len(query), dtype="float32").reshape(-1, 1)
        query_nhsw_vectors = np.hstack((query, aux_dim))
    else:
        query_nhsw_vectors = query
    D, I = index.search(query_nhsw_vectors, k) # distance, index
    return D, I

# https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/rag/retrieval_rag.py#L188
def get_top_docs(index, embeddings: np.ndarray, n_docs=5, aux_dim=False) -> Tuple[np.ndarray, np.ndarray]:
    if aux_dim:
        aux_dim = np.zeros(len(embeddings), dtype="float32").reshape(-1, 1)
        query_nhsw_vectors = np.hstack((embeddings.detach().numpy(), aux_dim))
    else:
        query_nhsw_vectors = embeddings.unsqueeze(0).cpu().detach().numpy()
    _, docs_ids = index.search(query_nhsw_vectors, n_docs)
    vectors = [[index.reconstruct(int(doc_id)) for doc_id in doc_ids] for doc_ids in docs_ids]
    return docs_ids, np.array(vectors)

def add_vectors_to_index(index, vectors, aux_dim=False):
    if aux_dim:
        aux_dim = np.zeros(len(vectors), dtype="float32").reshape(-1, 1)
        vectors = np.hstack((vectors, aux_dim))
    index.add(vectors)

def build_index(index, documents, retriever, tokenizer, embed_dim=None, index_path=None):
    print("Building index...")
    total_docs = len(documents)
    for i, doc in enumerate(documents, start=1):
        encoded_doc = encode_document(doc, tokenizer)
        embeddings = embed(encoded_doc, retriever, matryoshka_dim=embed_dim)
        add_vectors_to_index(index, embeddings)
        print(f"Adding document {i}/{total_docs} to index...", end='\r')
    print("\nIndex building complete.")

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if index_path:
        write_index(index, index_path)
    


