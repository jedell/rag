import faiss
from .nomic import embed, encode_document

# TODO: look at FAISS impl here
# https://www.llamaindex.ai/open-source
# https://python.langchain.com/docs/modules/data_connection/vectorstores/

def init_index(embed_dim):
    print("Initializing index...")
    index = faiss.IndexFlatL2(embed_dim) #IndexHNSWFlat(embed_dim)
    return index

def add(index, embeddigns):
    index.add(embeddigns)

def search(index, query, k):
    D, I = index.search(query, k) # distance, index
    return D, I

def build_index(index, documents, retriever, tokenizer):
    print("Building index...")
    print("TODO: make this faster/save to db")
    for doc in documents:
        encoded_doc = encode_document(doc, tokenizer)
        embeddings = embed(encoded_doc, retriever)
        add(index, embeddings)


