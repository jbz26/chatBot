import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from typing import List
from fastapi import File, UploadFile, Form
from src.rag.vectorstore import VectorDB
from src.base.llm_model import get_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA
from src.chat.main import build_chat_chain
from src.rag.file_loader import Loader
# from langchain_experimental.text_splitter import SemanticChunker
# from src.rag.extractor import Extractor


llm = get_llm("gemini-2.0-flash", temperature=0.9)

genai_docs = "./data_source/generative_ai"

# --------- Chains----------------

genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

chat_chain = build_chat_chain(genai_chain, 
                              history_folder="./chat_histories",
                              max_history_length=6)


# --------- App - FastAPI ----------------

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

# --------- Routes - FastAPI ----------------

@app.get("/check")
async def check():
    return {"status": "ok"}


@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

@app.post("/upload")
async def upload(
    file: List[UploadFile] = File(None),
):
    sources = []

    if file:
        for f in file:
            file_path = f"./data_source/generative_ai/{f.filename}"
            with open(file_path, "wb") as out_file:
                content = await f.read()
                out_file.write(content)
            sources.append(file_path)


    # Load all files with multiprocessing-aware loader
    loader = Loader(file_type=f.filename.split(".")[-1],
                    split_kwargs={
                        "chunk_size": 300,
                        "chunk_overlap": 0
                    })
    docs = loader.load(sources, workers=7)

    return {"message": "Docs processed", "extracted": docs}


# --------- Langserve Routes - Playground ----------------
add_routes(app, 
           genai_chain, 
           path="/generative_ai")

add_routes(app,
           chat_chain,
           path="/chat")
