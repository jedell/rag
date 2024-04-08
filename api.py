from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn

app = FastAPI()

from retriever.index import init_index
from utils import load_documents, setup_model

model = None
documents = None

class ChatRequest(BaseModel):
    user_id: str
    prompt: str

class ChatResponse(BaseModel):
    user_id: str
    response: str
    context: Optional[str] = None

@app.on_event("startup")
def load_model():
    global model, documents
    documents_path = "data/chunks"
    index_path = "index/dune.index"
    embed_dim = 768

    index = init_index(embed_dim, index_path)
    documents = load_documents(documents_path)
    model = setup_model(index, documents)
    model.eval()
    print("Model and documents loaded successfully.")

@app.post("/chat/", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    user_input = chat_request.prompt
    if not user_input:
        raise HTTPException(status_code=400, detail="Prompt is required")

    context, compl = model.generate(user_input)
    response = ChatResponse(
        user_id=chat_request.user_id,
        response=compl,
        context=context
    )
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
