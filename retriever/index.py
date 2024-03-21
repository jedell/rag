import os
import numpy as np
import faiss
from typing import Tuple
from faiss import write_index, read_index
from .nomic import embed, encode_document

# TODO: look at FAISS impl here
# https://www.llamaindex.ai/open-source
# https://python.langchain.com/docs/modules/data_connection/vectorstores/

def init_index(embed_dim, index_path=None, M=16):
    print("Initializing index...")
    if index_path and os.path.exists(index_path):
        return read_index(index_path)
    index = faiss.IndexHNSWFlat(embed_dim + 1, M) # why + 1 ???
    return index

def search(index, query, k):
    D, I = index.search(query, k) # distance, index
    return D, I

# https://github.com/huggingface/transformers/blob/66ce9593fdb8e340df546ddd0774eb444f17a12c/src/transformers/models/rag/retrieval_rag.py#L188
def get_top_docs(index, embeddings: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray]:
    aux_dim = np.zeros(len(embeddings), dtype="float32").reshape(-1, 1)
    query_nhsw_vectors = np.hstack((embeddings, aux_dim))
    _, docs_ids = index.search(query_nhsw_vectors, n_docs)
    vectors = [[index.reconstruct(int(doc_id))[:-1] for doc_id in doc_ids] for doc_ids in docs_ids]
    return docs_ids, np.array(vectors)

def build_index(index, documents, retriever, tokenizer, index_path=None):
    print("Building index...")    
    for doc in documents:
        encoded_doc = encode_document(doc, tokenizer)
        embeddings = embed(encoded_doc, retriever)
        index.add(embeddings)

    if index_path:
        write_index(index, index_path)
    


