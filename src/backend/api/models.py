from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., example="What is the capital of Vietnam?")


class ChatResponse(BaseModel):
    response: str

class UploadFileRequest(BaseModel):
    file_paths: list[str]

class UploadFileRespones(BaseModel):
    status: str 
    new_doc_count: int
    
