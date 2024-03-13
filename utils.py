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
                        documents.extend(chunk_text(content, chunk_size))
                    else:
                        documents.append(content)
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                    if chunk_size is not None:
                        documents.extend(chunk_text(content, chunk_size))
                    else:
                        documents.append(content)
    elif os.path.isfile(path):
        # If path is a single file, read the file
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                if chunk_size is not None:
                    documents.extend(chunk_text(content, chunk_size))
                else:
                    documents.append(content)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='latin-1') as file:
                content = file.read()
                if chunk_size is not None:
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
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

