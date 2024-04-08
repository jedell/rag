from generate import generate
from retriever.index import init_index, build_index
from utils import load_documents
from utils import setup_model

import sys

def main(documents_path: str, index_path: str):

    embed_dim = 768

    index = init_index(embed_dim, index_path)
    # load documents into index
    documents = load_documents(documents_path)
    print("Documents loaded:", len(documents))

    model = setup_model(index, documents)
    model.eval()

    print("Enter your prompt or type 'exit' to quit:")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            sys.exit(0)

        context, compl = model.generate(user_input)
        
        print("Dune QA:", compl)
        print("Supporting Text:", context)
        print("\nEnter another prompt or type 'exit' to quit:")

main("data/chunks", "index/dune.index")

