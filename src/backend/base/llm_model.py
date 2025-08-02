# from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass
from dotenv import load_dotenv


load_dotenv()

# def get_llm(model: str = "gemini-2.0-flash", temperature: float = 0.0, max_tokens: int = None, **kwargs):
#     """
#     Initialize Gemini LLM via Google Generative AI using LangChain.
#     """
#     if not os.getenv("GOOGLE_API_KEY"):
#         if get_token():
#             os.environ["GOOGLE_API_KEY"] = get_token()
#         else:
#             os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key:")
#         print("Waiting API key from frontend...")
#     llm = ChatGoogleGenerativeAI(
#         model=model,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         timeout=None,
#         max_retries=2,
#         api_key=os.getenv("GOOGLE_API_KEY"),
#         **kwargs
#     )
#     return llm

def get_openAI_lookalike(model: str ="qwen/qwen3-coder:free" ,temperature: float = 0.0, max_tokens: int = None, api_base:str = "https://api.openai.com/v1/" , **kwargs):
    if not os.getenv("GOOGLE_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    else: 
        api_key = getpass.getpass("Enter your OpenAI API key:")
    llm=OpenAILike(model=model,
            api_key = "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",
            temperature = temperature,
            max_tokens = max_tokens,
            api_base = api_base,
            is_chat_model=True,
            is_function_calling_model=True, 
            **kwargs
    )
    return llm
    