from typing import Type, Union
from langchain_chroma import Chroma
# from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(
        self,
        documents=None,
        vector_db_class: Type[Chroma] = Chroma,
        embedding=HuggingFaceEmbeddings(),
    ) -> None:
        self.vector_db_class = vector_db_class
        self.embedding = embedding
        self.db = self._build_db(documents)

    def _build_db(self, documents):
        return self.vector_db_class.from_documents(documents=documents, embedding=self.embedding)

    def get_retriever(self, search_type="similarity", search_kwargs={"k": 10}):
        return self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

