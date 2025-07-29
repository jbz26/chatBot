from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from src.rag.file_loader import Loader
from src.chat.history import create_session_factory
from src.chat.output_parser import Str_OutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.rag.file_loader import Loader
from src.rag.vectorstore import VectorDB
from src.rag.offline_rag import Offline_RAG


# class InputQA(BaseModel):
#     question: str = Field(..., title="Question to ask the model")

class InputQA(BaseModel):
    human_input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "human_input"}},
    )


class OutputQA(BaseModel):
    answer: str = Field(..., title="Answer from the model")

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ]
)

class InputChat(BaseModel):
    human_input: str = Field(
        ...,
        description="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "human_input"}},
    )


# def build_rag_chain(llm, data_dir, data_type):
def build_rag_chain(llm, data_dir):
    # doc_loaded = Loader().load(file_paths=data_dir)
    loader = Loader(split_kwargs={
        "chunk_size": 300,
        "chunk_overlap": 0
    })
    doc_loaded = loader.load_directory(dir_path=data_dir, recursive=True)
    # doc_loaded = MyFileReader().load_data(load_dir=data_dir, workers=4)
    # print("______________________________________________")
    # print(f"Type of doc_loaded: {type(doc_loaded)}")
    retriever = VectorDB(documents = doc_loaded).get_retriever()
    # retriever = EnhancedVectorDB(documents=doc_loaded).get_retriever()
    rag_chain = Offline_RAG(llm).get_chain(retriever, history_folder="./chat_histories", max_history_length=6)
    return rag_chain

