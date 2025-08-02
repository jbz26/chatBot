from src.backend.api.models import UploadFileRequest, UploadFileRespones, ChatRequest, ChatResponse
from fastapi import APIRouter , Request, UploadFile, File
import os
import tempfile

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
    
    
@router.post("/add_files_directly", response_model=UploadFileRespones)
async def add_files(request: Request, file: UploadFile = File(...)):
    # Đọc nội dung file
    contents = await file.read()
    print(contents)
    chat_engine = request.app.state.chat_engine

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp_file:
        tmp_file.write(contents)
        tmp_file_path = tmp_file.name

    # xử lý
    new_docs = chat_engine.add_files([tmp_file_path])

    num_new_docs = len(new_docs)
    # xóa file sau khi xử lý
    os.remove(tmp_file_path)

    return {
        "status": "ok",
        "new_doc_count": num_new_docs
    }


