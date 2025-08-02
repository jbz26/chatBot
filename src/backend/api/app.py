import os
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.rag_langchain.chat_engine import ChatEngine
from backend.api.routes.doc_routes import router as doc_router
from backend.api.routes.chat_routes import router as chat_router

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chạy logic trước khi app bắt đầu nhận request
    print("🚀 App is starting up...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,api_key=api_key)
    embedding=OpenAIEmbeddings(api_key=api_key)
    app.state.chat_engine = ChatEngine(llm=llm,embedding=embedding,collection_name="my_file")
    yield
    # Chạy logic khi app sắp tắt
    print("🛑 App is shutting down...")
    
app = FastAPI(lifespan=lifespan)
app.include_router(doc_router, prefix="/documents", tags=["Documents"])
app.include_router(chat_router, prefix="/user",tags=["Chat"])

