import torch
from mistral.model import Transformer as Mistral
from mistral.tokenizer import Tokenizer
from generate import generate
from retriever.index import init_index, build_index
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from utils import load_documents

def main(generator_path: str, documents_path: str):

    embed_dim = 512

    generator_tokenizer = Tokenizer(str(Path(generator_path) / "tokenizer.model"))
    retriever_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)

    generator = Mistral.from_folder(Path(generator_path))
    retriever = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True, rotary_scaling_factor=2)

    generator.eval()
    retriever.eval()

    index = init_index(embed_dim)

    # load documents into index
    documents = load_documents(documents_path)

    build_index(index, documents, retriever, retriever_tokenizer)

    prompt = "What is the meaning of life?"

    compl, logprobs = generate([prompt], 
                              generator, generator_tokenizer, 
                              retriever, retriever_tokenizer, 
                              index)
    
    print(compl)

    




