import os
from typing import List


def load_documents(path: str, chunk_size: int = None) -> List[str]:
    """
    Load documents from a specified path and optionally chunk the text based on a specified chunk size.
    This can handle both a single document or a directory of documents.
    Each document is read as a string, optionally chunked, and returned in a list.
    """
    documents = []
    if os.path.isdir(path):
        # If path is a directory, iterate over all files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if chunk_size is not None:
                        print(f"Chunking {file_path}, chunk size: {chunk_size}")
                        documents.extend(chunk_text(content, chunk_size))
                    else:
                        documents.append(content)
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    if chunk_size is not None:
                        print(f"Chunking {file_path}, chunk size: {chunk_size}")
                        documents.extend(chunk_text(content, chunk_size))
                    else:
                        documents.append(content)
    elif os.path.isfile(path):
        # If path is a single file, read the file
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                if chunk_size is not None:
                    print(f"Chunking {path}, chunk size: {chunk_size}")
                    documents.extend(chunk_text(content, chunk_size))
                else:
                    documents.append(content)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as file:
                content = file.read()
                if chunk_size is not None:
                    print(f"Chunking {path}, chunk size: {chunk_size}")
                    documents.extend(chunk_text(content, chunk_size))
                else:
                    documents.append(content)
    else:
        raise ValueError(f"Path {path} is neither a valid file nor a directory.")
    
    return documents

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Divide a text into semantic pieces of a specified size. This is a naive implementation that simply chunks
    the text by a character count. More sophisticated methods could be used for semantic chunking.
    i.e https://python.langchain.com/docs/modules/data_connection/document_transformers/
    """
    # return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    # docs = text_splitter.create_documents([text])
    # print("Chunks:", len(docs))
    # return [doc.page_content for doc in docs]
    return text

if __name__ == "__main__":
    # documents = load_documents("data/Dune 1 Dune.txt", chunk_size='semantic')
    
    # # save chunks to separate files
    # os.makedirs("data/chunks/Dune 1 Dune", exist_ok=True)
    # for i, doc in enumerate(documents):
    #     with open(f"data/chunks/Dune 1 Dune/dune1_chunk_{i}.txt", "w") as f:
    #         f.write(doc)

    from transformers import AutoTokenizer

    retriever_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', model_max_length=8192)

    texts = [
        "Dune 1 Dune.txt",
        "Dune 2 Dune Messiah.txt",
        "Dune 3 Children of Dune.txt",
        "Dune 4 God Emperor.txt",
        "Dune 5 Heretics of Dune.txt",
        "Dune 6 Chapterhouse.txt",
    ]
    texts = [f"data/{text}" for text in texts]

    for out_idx, text_path in enumerate(texts):
        # split Dune 1 by '===' indicating chapters and further split chapters if needed
        with open(text_path, "r") as f:
            dune_text = f.read()
        chapters = dune_text.split("===")
        print(f"Processing {text_path}, len(chapters): {len(chapters)}")
        
        sub_chapters = []
        for idx, chapter in enumerate(chapters):
            tokens = retriever_tokenizer(chapter, return_tensors='pt')
            token_len = tokens['input_ids'].shape[1]
            
            if token_len > 2048:
                num_splits = 2 
                while True:
                    sub_chapter_len = len(chapter) // num_splits
                    sub_chapters_split = []
                    sub_chapter_char_lens = []
                    start = 0
                    while start < len(chapter):
                        end = start + sub_chapter_len
                        if end < len(chapter):

                            while end < len(chapter) and chapter[end] != '\n':
                                end += 1

                        sub_chapters_split.append(chapter[start:end])
                        sub_chapter_char_lens.append((start, end))
                        start = end
                        
                    sub_chapter_tokens = [retriever_tokenizer(sub_chapter, return_tensors='pt')['input_ids'].shape[1] for sub_chapter in sub_chapters_split]
                    if all(sub_len <= 2048 for sub_len in sub_chapter_tokens):
                        sub_chapter_texts = [chapter[start:end] for start, end in sub_chapter_char_lens]
                        if len(sub_chapter_texts) > num_splits:
                            sub_chapter_texts[-2] += sub_chapter_texts[-1]
                            sub_chapter_texts.pop(-1)
                        sub_chapters.extend(sub_chapter_texts)
                        break
                    else:
                        num_splits += 1
            else:
                sub_chapters.append(chapter)
        
        print(f"Total sub-chapters: {len(sub_chapters)}")

        # save sub-chapters to separate files
        os.makedirs(f"data/chunks/dune{out_idx+1}", exist_ok=True)
        # remove all from data/chunks/dune{out_idx+1}
        for file in os.listdir(f"data/chunks/dune{out_idx+1}"):
            os.remove(f"data/chunks/dune{out_idx+1}/{file}")

        for i, sub_chapter in enumerate(sub_chapters):
            with open(f"data/chunks/dune{out_idx+1}/dune{out_idx+1}_sub_chapter_{i+1}.txt", "w") as f:
                f.write(sub_chapter.strip().replace("\t", "").replace("    ", "").replace("\n\n", "\n"))


    

