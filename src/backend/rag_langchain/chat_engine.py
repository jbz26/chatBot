from langchain_chroma import Chroma
import chromadb
from langchain_core.runnables import  RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from backend.rag_langchain.file_loader import read_uploaded_files

class ChatEngine():
    def __init__(self, llm, embedding ,collection_name: str = "my_chat_db", chroma_db_path: str = "./chroma_db", max_length: int = 1024):
        self.chroma_db_path = chroma_db_path
        self.max_length = max_length
        self.llm = llm
        self.embedding = embedding
        self.collection_name = collection_name
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="question")
        self.prompt = None
        self.init_vector_store()
        self.get_prompt()
        self.get_or_create_chain()

    def init_vector_store(self):
        client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.vector_store = Chroma(client=client,
                                   embedding_function=self.embedding,
                                   collection_name=self.collection_name
                                   )

    def get_or_create_chain(self):
        retriever = self.vector_store.as_retriever()
        
        def format_output(question: str):
            context_docs = retriever.invoke(question)
            context_str = "\n\n".join([doc.page_content for doc in context_docs])
            return {
                "question" : question,
                "context": context_str
            }
            

        self.rag_chain  = (
           RunnableLambda(format_output)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return self.rag_chain
    
    def get_prompt(self):
        # Prompt
        if not self.prompt:
            template = """Answer the question based only on the following context:
            {context}

            Question: {question}
            """

            self.prompt = ChatPromptTemplate.from_template(template)
        return self.prompt
    
    def add_files(self,file_paths: list[str]):
        documents = read_uploaded_files(file_paths)
        new_docs = self.vector_store.add_documents(documents)
        self.get_or_create_chain()
        return new_docs
    def chat(self, input:str) -> str:
        response = self.rag_chain.invoke(input)
        return response
    
        

        
        