from src.backend.api.models import ChatRequest, ChatResponse
from fastapi import APIRouter , Request

router = APIRouter(tags=["Chat"])

@router.post("/chat",response_model=ChatResponse)
def chat_with_model(req: ChatRequest, request: Request):
    chat_engine = request.app.state.chat_engine
    response = chat_engine.chat(req.message)
    return {"response": response}