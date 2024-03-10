from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from mistral.model import Transformer
from mistral.generate import generate as generate_response
from transformers import AutoTokenizer, AutoModel
from retriever import nomic
from retriever.index import search
import faiss

def generate(
        query, 
        generator: Transformer, 
        tokenizer: Tokenizer,
        retriever: AutoModel,
        retriever_tokenizer: AutoTokenizer,
        index: faiss.IndexHNSWFlat
        ):
    
    encoded_query = nomic.encode_query(query, retriever_tokenizer)
    embeded_query = nomic.embed(encoded_query, retriever)

    context = search(index, embeded_query, k=1)

    prompt = f"""{context} <s>[INST] {query} [/INST]"""

    completion, logprobs = generate_response([prompt], generator, tokenizer)

    return completion, logprobs
