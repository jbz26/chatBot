import re
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable

# Assuming these imports exist in your project
# from src.chat.history import create_session_factory

# Setup logging
logger = logging.getLogger(__name__)


class EnhancedStrOutputParser(StrOutputParser):
    """Enhanced string output parser with multiple extraction patterns and fallback handling."""
    
    def __init__(self, 
                 primary_pattern: str = r"Answer:\s*(.*)",
                 fallback_patterns: List[str] = None,
                 clean_output: bool = True) -> None:
        super().__init__()
        self.primary_pattern = primary_pattern
        self.fallback_patterns = fallback_patterns or [
            r"Response:\s*(.*)",
            r"Result:\s*(.*)",
            r"Output:\s*(.*)",
        ]
        self.clean_output = clean_output
    
    def parse(self, text: str) -> str:
        """Parse the text response and extract the answer."""
        if not isinstance(text, str):
            text = str(text)
        
        return self.extract_answer(text)
    
    def extract_answer(self, text_response: str) -> str:
        """
        Extract answer from text response using multiple patterns.
        
        Args:
            text_response: The raw text response from the LLM
            
        Returns:
            Extracted answer text or original text if no pattern matches
        """
        # Try primary pattern first
        match = re.search(self.primary_pattern, text_response, re.DOTALL | re.IGNORECASE)
        if match:
            answer_text = match.group(1).strip()
            return self._clean_text(answer_text) if self.clean_output else answer_text
        
        # Try fallback patterns
        for pattern in self.fallback_patterns:
            match = re.search(pattern, text_response, re.DOTALL | re.IGNORECASE)
            if match:
                answer_text = match.group(1).strip()
                logger.debug(f"Used fallback pattern: {pattern}")
                return self._clean_text(answer_text) if self.clean_output else answer_text
        
        # If no pattern matches, return cleaned original text
        logger.debug("No extraction pattern matched, returning original text")
        return self._clean_text(text_response) if self.clean_output else text_response
    
    def _clean_text(self, text: str) -> str:
        """Clean the extracted text by removing extra whitespace and formatting."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Remove common artifacts
        text = re.sub(r'^[\-\*\•]\s*', '', text)  # Remove leading bullets/dashes
        return text


class DocumentFormatter:
    """Handles different document formatting strategies for RAG context."""
    
    def __init__(self, 
                 format_type: str = "simple",
                 max_docs: Optional[int] = None,
                 include_metadata: bool = False,
                 separator: str = "\n\n"):
        self.format_type = format_type
        self.max_docs = max_docs
        self.include_metadata = include_metadata
        self.separator = separator
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format documents based on the specified strategy."""
        if not docs:
            return "No relevant documents found."
        
        # Limit number of documents if specified
        if self.max_docs:
            docs = docs[:self.max_docs]
        
        if self.format_type == "simple":
            return self._format_simple(docs)
        elif self.format_type == "numbered":
            return self._format_numbered(docs)
        elif self.format_type == "with_source":
            return self._format_with_source(docs)
        elif self.format_type == "detailed":
            return self._format_detailed(docs)
        else:
            return self._format_simple(docs)
    
    def _format_simple(self, docs: List[Document]) -> str:
        """Simple formatting - just content separated by newlines."""
        return self.separator.join(doc.page_content for doc in docs)
    
    def _format_numbered(self, docs: List[Document]) -> str:
        """Numbered formatting with document indices."""
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            formatted_docs.append(f"Document {i}:\n{doc.page_content}")
        return self.separator.join(formatted_docs)
    
    def _format_with_source(self, docs: List[Document]) -> str:
        """Format with source information if available."""
        formatted_docs = []
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown source')
            formatted_docs.append(f"Source: {source}\n{content}")
        return self.separator.join(formatted_docs)
    
    def _format_detailed(self, docs: List[Document]) -> str:
        """Detailed formatting with all available metadata."""
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = f"Document {i}:\n{doc.page_content}"
            if self.include_metadata and doc.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                content = f"Document {i} ({metadata_str}):\n{doc.page_content}"
            formatted_docs.append(content)
        return self.separator.join(formatted_docs)


class AdvancedRAG:
    """
    Advanced RAG system with improved error handling, customizable prompts,
    and multiple formatting options.
    """
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 prompt_template: Optional[Union[str, ChatPromptTemplate]] = None,
                 output_parser: Optional[StrOutputParser] = None,
                 document_formatter: Optional[DocumentFormatter] = None):
        """
        Initialize the Advanced RAG system.
        
        Args:
            llm: Language model to use
            prompt_template: Custom prompt template or hub prompt name
            output_parser: Custom output parser
            document_formatter: Custom document formatter
        """
        self.llm = llm
        self.output_parser = output_parser or EnhancedStrOutputParser()
        self.document_formatter = document_formatter or DocumentFormatter()
        
        # Set up prompt
        if isinstance(prompt_template, str):
            try:
                self.prompt = hub.pull(prompt_template)
            except Exception as e:
                logger.error(f"Failed to pull prompt from hub: {e}")
                self.prompt = self._get_default_prompt()
        elif isinstance(prompt_template, ChatPromptTemplate):
            self.prompt = prompt_template
        elif prompt_template is None:
            try:
                self.prompt = hub.pull("rlm/rag-prompt")
            except Exception as e:
                logger.warning(f"Failed to pull default prompt from hub: {e}, using fallback")
                self.prompt = self._get_default_prompt()
        else:
            raise ValueError("prompt_template must be a string (hub name) or ChatPromptTemplate")
    
    def _get_default_prompt(self) -> ChatPromptTemplate:
        """Get a default RAG prompt template."""
        template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        Context: {context}

        Question: {question}

        Answer:"""
        return ChatPromptTemplate.from_template(template)
    
    def get_chain(self, 
                  retriever: BaseRetriever,
                  history_folder: str = "./chat_histories",
                  max_history_length: int = 6,
                  enable_history: bool = True) -> Runnable:
        """
        Create the RAG chain with optional conversation history.
        
        Args:
            retriever: Document retriever
            history_folder: Folder to store chat histories
            max_history_length: Maximum number of messages to keep in history
            enable_history: Whether to enable conversation history
            
        Returns:
            Configured RAG chain
        """
        # Input processing with better error handling
        def safe_string_conversion(x):
            """Safely convert input to string."""
            if hasattr(x, "to_string"):
                return x.to_string()
            elif isinstance(x, dict) and "question" in x:
                return x["question"]
            elif isinstance(x, dict) and "input" in x:
                return x["input"]
            else:
                return str(x)
        
        # Create input data mapping
        input_data = {
            "context": retriever | RunnableLambda(self.document_formatter.format_docs),
            "question": RunnablePassthrough() | RunnableLambda(safe_string_conversion)
        }
        
        # Add human_input for history if needed
        if enable_history:
            input_data["human_input"] = RunnablePassthrough() | RunnableLambda(safe_string_conversion)
        
        # Create the basic RAG chain
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.output_parser
        )
        
        # Add conversation history if enabled
        if enable_history:
            try:
                # Import here to avoid circular imports
                from src.chat.history import create_session_factory
                
                chain_with_history = RunnableWithMessageHistory(
                    rag_chain,
                    create_session_factory(
                        base_dir=history_folder,
                        max_history_length=max_history_length
                    ),
                    input_messages_key="human_input",
                    history_messages_key="chat_history",
                )
                return chain_with_history
            except ImportError as e:
                logger.warning(f"History module not available: {e}. Returning chain without history.")
                return rag_chain
        
        return rag_chain
    
    def create_simple_chain(self, retriever: BaseRetriever) -> Runnable:
        """Create a simple RAG chain without conversation history."""
        return self.get_chain(retriever, enable_history=False)
    
    def invoke_with_sources(self, 
                           chain: Runnable, 
                           query: str, 
                           session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Invoke the chain and return both answer and source documents.
        
        Args:
            chain: The RAG chain
            query: User question
            session_id: Session ID for history (if applicable)
            
        Returns:
            Dictionary with answer and source information
        """
        try:
            # Prepare input
            chain_input = {"question": query}
            if session_id:
                chain_input = {"question": query, "session_id": session_id}
            
            # Get answer
            if session_id and hasattr(chain, 'invoke'):
                answer = chain.invoke(
                    chain_input,
                    config={"configurable": {"session_id": session_id}}
                )
            else:
                answer = chain.invoke(chain_input)
            
            return {
                "answer": answer,
                "query": query,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error invoking chain: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "query": query,
                "session_id": session_id,
                "error": str(e)
            }


class RAGBuilder:
    """Builder class for creating customized RAG systems."""
    
    def __init__(self):
        self.llm = None
        self.prompt_template = None
        self.output_parser = None
        self.document_formatter = None
    
    def with_llm(self, llm: BaseLanguageModel):
        """Set the language model."""
        self.llm = llm
        return self
    
    def with_prompt(self, prompt_template: Union[str, ChatPromptTemplate]):
        """Set the prompt template."""
        self.prompt_template = prompt_template
        return self
    
    def with_output_parser(self, 
                          primary_pattern: str = r"Answer:\s*(.*)",
                          fallback_patterns: List[str] = None,
                          clean_output: bool = True):
        """Set the output parser configuration."""
        self.output_parser = EnhancedStrOutputParser(
            primary_pattern=primary_pattern,
            fallback_patterns=fallback_patterns,
            clean_output=clean_output
        )
        return self
    
    def with_document_formatter(self,
                               format_type: str = "simple",
                               max_docs: Optional[int] = None,
                               include_metadata: bool = False):
        """Set the document formatter configuration."""
        self.document_formatter = DocumentFormatter(
            format_type=format_type,
            max_docs=max_docs,
            include_metadata=include_metadata
        )
        return self
    
    def build(self) -> AdvancedRAG:
        """Build the RAG system."""
        if self.llm is None:
            raise ValueError("LLM must be specified")
        
        return AdvancedRAG(
            llm=self.llm,
            prompt_template=self.prompt_template,
            output_parser=self.output_parser,
            document_formatter=self.document_formatter
        )


# Convenience functions for quick setup
def create_simple_rag(llm: BaseLanguageModel, 
                     retriever: BaseRetriever) -> Runnable:
    """Create a simple RAG chain without history."""
    rag = AdvancedRAG(llm)
    return rag.create_simple_chain(retriever)


def create_rag_with_history(llm: BaseLanguageModel,
                           retriever: BaseRetriever,
                           history_folder: str = "./chat_histories") -> Runnable:
    """Create a RAG chain with conversation history."""
    rag = AdvancedRAG(llm)
    return rag.get_chain(retriever, history_folder=history_folder)


# Example usage
if __name__ == "__main__":
    # Example of using the builder pattern
    """
    rag = (RAGBuilder()
           .with_llm(your_llm)
           .with_document_formatter(format_type="numbered", max_docs=5)
           .with_output_parser(primary_pattern=r"Response:\s*(.*)")
           .build())
    
    chain = rag.get_chain(your_retriever)
    result = rag.invoke_with_sources(chain, "What is the capital of France?")
    print(result["answer"])
    """
    pass