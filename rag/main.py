import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag.file_loader import Loader
from rag.vectorstore import VectorDB
from rag.offline_rag import Offline_RAG

from src.base.llm_model import get_llm

from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def build_chain(llm, data_dir, data_type):
    doc_loaded = Loader(file_type=data_type).load_dir(data_dir, workers=2)
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    rag_chain = Offline_RAG(llm).get_chain(retriever)
    return rag_chain


# --------- RAG-Docs ----------------
genai_docs = "./data_source/generative_ai"
ml_docs = "./data_source/machine_learning"

# --------- RAG-LLM ----------------
llm = get_llm()

# --------- RAG-Chain ----------------
genai_chain = build_chain(llm, data_dir=genai_docs, data_type="pdf")
ml_chain = build_chain(llm, data_dir=ml_docs, data_type="html")

# --------- FastAPI Routes ----------------

class InputQuestion(BaseModel):
    question: str = Field(..., title="Question to ask the model")

class OutputAnswer(BaseModel):
    answer: str = Field(..., title="Answer from the model")


@app.get("/check")
async def check():
    return {"status": "ok"}


@app.post("/generative_ai", response_model=OutputAnswer)
async def generative_ai(inputs: InputQuestion):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}


@app.post("/machine_learning", response_model=OutputAnswer)
async def machine_learning(inputs: InputQuestion):
    answer = ml_chain.invoke(inputs.question)
    return {"answer": answer}

# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           genai_chain, 
           path="/generative_ai")

add_routes(app, 
           ml_chain, 
           path="/machine_learning")
