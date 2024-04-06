from generate import generate
from retriever.index import init_index, build_index
from utils import load_documents
from utils import setup_model

def main(generator_path: str, documents_path: str, index_path: str):

    embed_dim = 768

    index = init_index(embed_dim, index_path)
    model = setup_model(index)
    model.eval()

    # load documents into index
    documents = load_documents(documents_path)
    print("Chunks len", len(documents))

    # build_index(index, documents, retriever, retriever_tokenizer, index_path=index_path)
    prompt = "What is the meaning of life?"

    context, compl = model.generate(prompt)
    
    print("Context:", context)
    print("Prompt:", prompt)
    print("Completion:", compl)

main("_model", "data/chunks", "index/dune.index")


