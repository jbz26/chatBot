import os
from contextlib import asynccontextmanager
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.backend.rag_langchain.chat_engine import ChatEngine
from src.backend.api.routes.doc_routes import router as doc_router
from src.backend.api.routes.chat_routes import router as chat_router
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "2")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chạy logic trước khi app bắt đầu nhận request
    print("🚀 App is starting up...")
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,api_key=api_key)
    llm=ChatOpenAI(model="qwen/qwen3-coder:free",
            api_key = "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",
            temperature = 0,
            base_url = "https://openrouter.ai/api/v1",
    )
    #embedding=OpenAIEmbeddings(api_key=api_key)
    embedding = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    app.state.chat_engine = ChatEngine(llm=llm,embedding=embedding,collection_name="my_file")
    yield
    # Chạy logic khi app sắp tắt
    print("🛑 App is shutting down...")
    
app = FastAPI(lifespan=lifespan)
app.include_router(doc_router, prefix="/documents", tags=["Documents"])
app.include_router(chat_router, prefix="/user",tags=["Chat"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ định domain cụ thể như ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # hoặc ["POST", "GET", "OPTIONS"]
    allow_headers=["*"],
)
