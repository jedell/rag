from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from mistral.model import Transformer
from mistral.generate import generate as generate_response
from transformers import AutoTokenizer, AutoModel
from retriever import nomic
from retriever.index import search
import faiss
from typing import List

def generate(
        query, 
        generator: Transformer, 
        tokenizer: Tokenizer,
        retriever: AutoModel,
        retriever_tokenizer: AutoTokenizer,
        index: faiss.IndexHNSWFlat,
        documents: List[str]
        ):
    
    encoded_query = nomic.encode_query(query[0], retriever_tokenizer)
    embeded_query = nomic.embed(encoded_query, retriever)

    D, I = search(index, embeded_query, k=1)
    context = [documents[i] for i in I.tolist()[0]]

    print(context)

    prompt = f"""{context} <s>[INST] {query} [/INST]"""

    completion, logprobs = generate_response([prompt], generator, tokenizer, max_tokens=512, temperature=0.1)

    return completion, logprobs, context, D
