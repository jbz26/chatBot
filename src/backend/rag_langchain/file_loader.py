from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


def read_uploaded_files(file_paths:list[str]) -> list[Document] | None:
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    for file_path in file_paths:
        filename = file_path.name if hasattr(file_path, "name") else str(file_path)
        ext = filename.lower().split('.')[-1]
        document = []
        
        if ext == "pdf":
            pdf_loader = PyPDFLoader(file_path)
            for page in pdf_loader.lazy_load():
                document.append(page)
        elif ext == "docx":
            docx_loader = Docx2txtLoader(file_path)
            for page in docx_loader.lazy_load():
                document.append(page)
        elif ext == "doc":
            doc_loader = UnstructuredWordDocumentLoader(file_path)
            for page in doc_loader.lazy_load():
                document.append(page)
        elif ext in ["xls", "xlsx"]:
            excel_loader = UnstructuredExcelLoader(file_path)
            for page in excel_loader.lazy_load():
                document.append(page)
        elif ext =="txt":
            txt_loader = TextLoader(file_path)
            for page in txt_loader.lazy_load:
                document.append(page)
                
        if document:
            split_documents = text_splitter.split_documents(document)
            documents.extend(split_documents)
            
    return documents