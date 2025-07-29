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
        
# chat_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
#         MessagesPlaceholder(variable_name="chat_histories"),
#         ("context", "{context}"),
#         ("human", "{human_input}"),
#     ]
# )
def get_default_prompt() -> ChatPromptTemplate:
    """Get a default RAG prompt template."""
    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Context: {context}

    Question: {question}

    Answer:"""
    return ChatPromptTemplate.from_template(template)

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        # self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt = get_default_prompt()
        self.str_parser = Str_OutputParser()
        

    
    # def get_chain(self, retriever, history_folder="./chat_histories", max_history_length=6):

    #     input_data = {
    #         "context": retriever,
    #         "question": RunnableLambda(lambda x: (
    #             x["human_input"] if isinstance(x["human_input"], str)
    #             else str(x["human_input"].get("question", "")) if isinstance(x["human_input"], dict)
    #             else str(x["human_input"])
    #         )),
    #     }
    #     # input_data["human_input"] = RunnablePassthrough() | RunnableLambda(Str_OutputParser().extract_answer)
    #     rag_chain = (
    #         input_data
    #         | self.prompt
    #         | self.llm
    #         | self.str_parser
    #     )
        
    #     chain_with_history = RunnableWithMessageHistory(
    #         rag_chain,
    #         create_session_factory(base_dir=history_folder, 
    #                                max_history_length=max_history_length),
    #         input_messages_key="human_input",
    #         history_messages_key="chat_history",
    #     )
    #     return chain_with_history
    def get_chain(self, retriever, history_folder="./chat_histories", max_history_length=6):
        input_data = {
            "context": RunnableLambda(lambda x: retriever.invoke(str(x["human_input"]))) | RunnableLambda(self.format_docs),
            "question": RunnableLambda(lambda x: str(x["human_input"]))
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            create_session_factory(base_dir=history_folder, max_history_length=max_history_length),
            input_messages_key="human_input",
            history_messages_key="chat_histories",
        )
        return chain_with_history



    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    