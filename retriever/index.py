import os
import faiss
from faiss import write_index, read_index
from .nomic import embed, encode_document

# TODO: look at FAISS impl here
# https://www.llamaindex.ai/open-source
# https://python.langchain.com/docs/modules/data_connection/vectorstores/

def init_index(embed_dim, index_path=None):
    print("Initializing index...")
    if index_path and os.path.exists(index_path):
        return read_index(index_path)
    index = faiss.IndexFlatL2(embed_dim) #IndexHNSWFlat(embed_dim)
    return index

def add(index, embeddigns):
    index.add(embeddigns)

def search(index, query, k):
    D, I = index.search(query, k) # distance, index
    return D, I

def build_index(index, documents, retriever, tokenizer, index_path=None):
    print("Building index...")    
    for doc in documents:
        encoded_doc = encode_document(doc, tokenizer)
        embeddings = embed(encoded_doc, retriever)
        add(index, embeddings)

    if index_path:
        write_index(index, index_path)
    


