from generate import generate
from retriever.index import init_index, build_index
from utils import load_documents
from utils import setup_model

def main(generator_path: str, documents_path: str, index_path: str):

    embed_dim = 768

    index = init_index(embed_dim, index_path)
        # load documents into index
    documents = load_documents(documents_path)
    print("Chunks len", len(documents))

    model = setup_model(index, documents)
    model.eval()


    # build_index(index, documents, retriever, retriever_tokenizer, index_path=index_path)
    prompt = "Why did the Fremin decide to follow Paul as their leader?"

    context, compl = model.generate(prompt)
    
    print("Context:", context)
    print("Prompt:", prompt)
    print("Completion:", compl)

main("_model", "data/chunks", "index/dune.index")


