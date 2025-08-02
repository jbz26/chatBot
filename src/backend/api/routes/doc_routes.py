from backend.api.models import UploadFileRequest, UploadFileRespones, ChatRequest, ChatResponse
from fastapi import APIRouter , Request

router = APIRouter(tags=["Documents"])
@router.get("/check")
def health_check():
    return {"status": "ok"}

@router.post("/add_files", response_model=UploadFileRespones)
def add_files(req: UploadFileRequest, request: Request):
    chat_engine = request.app.state.chat_engine
    new_docs = chat_engine.add_files(req.file_paths)
    num_new_docs = len(new_docs)
    return {"status": "ok", 
            "new_doc_count": num_new_docs
            }



