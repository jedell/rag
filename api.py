from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from retriever.index import init_index
from utils import load_documents, setup_model

model = None
documents = None

# In-memory database to store user conversations
conversations = {}

class ChatRequest(BaseModel):
    user_id: str
    prompt: str

class ChatResponse(BaseModel):
    user_id: str
    response: str
    context: Optional[str] = None

class CreateChatRequest(BaseModel):
    user_id: str

class Conversation(BaseModel):
    user_id: str
    chat_history: List[ChatResponse]


class MockModel:
    def generate(self, user_input):
        return "Fake context", "Fake response"

@app.on_event("startup")
def load_model():
    global model, documents
    documents_path = "data/chunks"
    index_path = "index/dune.index"
    embed_dim = 768

    # index = init_index(embed_dim, index_path)
    # documents = load_documents(documents_path)
    # model = setup_model(index, documents)
    # model.eval()
    # print("Model and documents loaded successfully.")

    # create fake model with fake generate function
    model = MockModel()

@app.put("/chat/", response_model=str)
def create_chat(create_chat_request: CreateChatRequest):
    chat_id = str(uuid.uuid4())
    conversations[chat_id] = Conversation(user_id=create_chat_request.user_id, chat_history=[])
    return chat_id


@app.post("/chat/{chat_id}", response_model=ChatResponse)
def chat(chat_id: str, chat_request: ChatRequest):
    if chat_id not in conversations:
        raise HTTPException(status_code=404, detail="Chat ID not found")
    
    user_input = chat_request.prompt
    if not user_input:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # TODO use conversation history to improve context
    context, compl = model.generate(user_input)
    response = ChatResponse(
        user_id=chat_request.user_id,
        response=compl,
        context=context
    )
    
    # Save the chat to the in-memory database
    conversations[chat_id].chat_history.append(response)
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

