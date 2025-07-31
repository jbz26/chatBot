import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.chat.history import create_session_factory
from src.chat.output_parser import Str_OutputParser


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    
    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response
        
# chat_prompt = ChatPromptTemplate.from_templete(
#     [
#         ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("context", "{context}"),
#         ("human", "{human_input}"),
#     ]
# )
def get_rag_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are generating a query for a similarity search to retrieve relevant documents. "
         "Based on the provided context and chat history, rephrase the user's question into a concise query that includes synonyms, related terms, or possible interpretations to maximize retrieval effectiveness. "
         "Output only the rephrased query or the error message as a single sentence.\n\nContext:\n{context}\n\nExample: Input: 'Is BERT a Bidirectional Encoder Representations from Transformers model?' -> Output: 'BERT, Bidirectional Encoder Representations from Transformers, language model, NLP'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ])

def get_clear_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", 
         "You are rephrasing a user's question to make it clear and specific for another model. "
         "Always rephrase the question concisely as a single sentence ending with a question mark, using the provided context and chat history to improve clarity, without answering it. "
         "Output only the rephrased question or the error message.\n\nContext:\n{context}\n\nExample: Input: 'Is BERT a Bidirectional Encoder Representations from Transformers model?' -> Output: 'Does BERT use bidirectional encoder representations from transformers according to the provided context?'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{human_input}"),
    ])

def get_default_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", 
         "Do not rephrase the question. Do not rephrase the question. Do not rephrase the question. Use the following context to answer the user's question. "
         "If you're unsure, say 'I don't know'.\n\nContext:\n{context}\n\nExample: Input: 'Does BERT use bidirectional encoder representations from transformers according to the provided context?' -> Output: 'Yes, BERT is a Bidirectional Encoder Representations from Transformers model.'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = get_default_prompt()
        self.str_parser = Str_OutputParser()

    def get_chain(self, retriever, history_folder="./chat_history", max_history_length=6):
        input_to_str = RunnableLambda(lambda x: (
            x["human_input"] if isinstance(x["human_input"], str)
            else str(x["human_input"][0]) if isinstance(x["human_input"], list) and len(x["human_input"]) > 0
            else str(x["human_input"])
        ))

        initial_retrieval = input_to_str | retriever | RunnableLambda(self.format_docs)

        rag_prompt_input = {
            "context": initial_retrieval,
            "human_input": input_to_str,
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        get_context = (
            rag_prompt_input
            | get_rag_prompt()
            | RunnableLambda(lambda x: print(f"RAG prompt input: {x}") or x) 
            | self.llm
            | self.str_parser
            | RunnableLambda(lambda x: print(f"Retrieval query: {x}") or x)  
        )

        preliminary_retrieval = get_context | retriever | RunnableLambda(self.format_docs)

        clear_prompt_input = {
            "context": preliminary_retrieval,
            "human_input": input_to_str,
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }
        clear_prompt = (
            get_clear_prompt()
            | RunnableLambda(lambda x: print(f"Clear prompt input: {x}") or x) 
            | self.llm
            | self.str_parser
            | RunnableLambda(lambda x: print(f"Clarified question: {x}") or x) 
        )

        clarified_question = clear_prompt_input | clear_prompt
        retrieval_pipeline = (
            RunnablePassthrough()
            | clarified_question
            | RunnableLambda(lambda x: print(f"Retriever input: {x}") or x)  # Debug
            | retriever
            | RunnableLambda(self.format_docs)
        )

        input_data = {
            "context": retrieval_pipeline,
            "question": clarified_question | RunnableLambda(lambda q: {"question": q}),
            "chat_history": RunnableLambda(lambda x: x.get("chat_history", [])),
        }

        rag_chain = (
            input_data
            | RunnableLambda(lambda x: print(f"Input data: {x}") or x)  
            | self.prompt
            | RunnableLambda(lambda x: print(f"Prompt input: {x}") or x)  
            | self.llm
            | RunnableLambda(lambda x: print(f"LLM output: {x}") or x)  
            | self.str_parser
            | RunnableLambda(lambda x: print(f"Final output: {x}") or x)  
        )

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            create_session_factory(base_dir=history_folder, max_history_length=max_history_length),
            input_messages_key="human_input",
            history_messages_key="chat_history",
        )
        return chain_with_history

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)