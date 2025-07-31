from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from typing import Union
from src.base.get_token import get_gemini_token, get_openai_token, get_deepseek_token
import os
import getpass
import re


def get_llm(model: str = "gemini-2.5-flash", temperature: float = 0.4, max_tokens: int = None, **kwargs):
    """
    Initialize Gemini LLM via Google Generative AI using LangChain.
    """
    # if not os.getenv("GOOGLE_API_KEY"):
    #     if get_token():
    #         os.environ["GOOGLE_API_KEY"] = get_token()
    #     else:
    #         os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key:")
    #         print("Waiting API key from frontend...")
    if  re.match(r"gemini-*", model):
        api_key = get_gemini_token() or os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2,
            api_key=api_key,
            **kwargs
        )
    elif re.match(r"gpt-*", model):
        api_key = get_openai_token() or os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2,
            api_key=api_key,
            **kwargs
        )
    elif re.match(r"deepseek-*", model).group(0):
        api_key = get_deepseek_token() or os.getenv("DEEPISEEK_API_KEY")
        llm = ChatDeepSeek(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=None,
            max_retries=2,
            api_key=api_key,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model: {model}")
    
    # llm = ChatGoogleGenerativeAI(
    #     model=model,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     timeout=None,
    #     max_retries=2,
    #     api_key=api_key,
    #     **kwargs
    # )
    return llm