from typing import Union, List, Dict, Optional
import glob
import os
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from langchain_community.document_loaders import (
    PyPDFLoader, 
    BSHTMLLoader, 
    UnstructuredExcelLoader, 
    Docx2txtLoader, 
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported file extensions mapping
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.html': 'html',
    '.htm': 'html',
    '.docx': 'docx',
    '.xlsx': 'xlsx',
    '.xls': 'xlsx',
    '.txt': 'txt',
    '.md': 'txt',
    '.csv': 'txt'
}


def remove_non_utf8_characters(text: str) -> str:
    """Remove non-UTF8 characters from text."""
    return ''.join(char for char in text if ord(char) < 128)


def safe_load_document(loader_func, file_path: str):
    """Safely load a document with error handling."""
    try:
        docs = loader_func(file_path)
        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return docs, None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return [], str(e)


def load_pdf(pdf_file: str):
    """Load PDF file."""
    return PyPDFLoader(pdf_file, extract_images=True).load()


def load_html(html_file: str):
    """Load HTML file."""
    return BSHTMLLoader(html_file).load()


def load_docx(docx_file: str):
    """Load DOCX file."""
    return Docx2txtLoader(docx_file).load()


def load_xlsx(xlsx_file: str):
    """Load Excel file."""
    return UnstructuredExcelLoader(xlsx_file).load()


def load_txt(txt_file: str):
    """Load text file."""
    return TextLoader(txt_file, encoding='utf-8').load()


def get_num_cpu() -> int:
    """Get number of CPU cores."""
    return multiprocessing.cpu_count()


def get_file_extension(file_path: str) -> str:
    """Get file extension in lowercase."""
    return Path(file_path).suffix.lower()


def validate_file_type(file_path: str) -> str:
    """Validate and return file type."""
    ext = get_file_extension(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported types: {list(SUPPORTED_EXTENSIONS.keys())}")
    return SUPPORTED_EXTENSIONS[ext]


class BaseLoader:
    """Base class for document loaders."""
    
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()
        self.loader_map = {
            'pdf': load_pdf,
            'html': load_html,
            'docx': load_docx,
            'xlsx': load_xlsx,
            'txt': load_txt
        }

    def load_files_parallel(self, files: List[str], file_type: str, workers: int = None) -> List:
        """Load multiple files in parallel with improved error handling."""
        if not files:
            return []
            
        workers = min(workers or self.num_processes, len(files))
        loader_func = self.loader_map[file_type]
        
        all_docs = []
        failed_files = []
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(safe_load_document, loader_func, file_path): file_path 
                for file_path in files
            }
            
            # Process results with progress bar
            with tqdm(total=len(files), desc=f"Loading {file_type.upper()} files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        docs, error = future.result()
                        if error:
                            failed_files.append((file_path, error))
                        else:
                            all_docs.extend(docs)
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
                        logger.error(f"Unexpected error processing {file_path}: {e}")
                    
                    pbar.update(1)
        
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files:")
            for file_path, error in failed_files:
                logger.warning(f"  {file_path}: {error}")
        
        logger.info(f"Successfully loaded {len(all_docs)} documents from {len(files) - len(failed_files)} files")
        return all_docs


class DocumentLoader(BaseLoader):
    """Main document loader class supporting multiple file types."""
    
    def __init__(self) -> None:
        super().__init__()

    def load_single_type(self, files: List[str], file_type: str, workers: int = None) -> List:
        """Load files of a single type."""
        return self.load_files_parallel(files, file_type, workers)

    def load_mixed_types(self, files: List[str], workers: int = None) -> List:
        """Load files of mixed types by grouping them."""
        # Group files by type
        files_by_type = {}
        for file_path in files:
            try:
                file_type = validate_file_type(file_path)
                if file_type not in files_by_type:
                    files_by_type[file_type] = []
                files_by_type[file_type].append(file_path)
            except ValueError as e:
                logger.warning(f"Skipping {file_path}: {e}")
        
        # Load each type separately
        all_docs = []
        for file_type, type_files in files_by_type.items():
            docs = self.load_files_parallel(type_files, file_type, workers)
            all_docs.extend(docs)
        
        return all_docs


class TextSplitter:
    """Enhanced text splitter with configurable options."""
    
    def __init__(self, 
                 separators: List[str] = None,
                 chunk_size: int = 300,
                 chunk_overlap: int = 0,
                 length_function: callable = len,
                 is_separator_regex: bool = False) -> None:
        
        if separators is None:
            separators = ['\n\n', '\n', ' ', '']
            
        self.splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex
        )
    
    def split_documents(self, documents):
        """Split documents into chunks."""
        if not documents:
            logger.warning("No documents to split")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks


class Loader:
    """Main loader class that combines loading and splitting functionality."""
    
    def __init__(self, 
                 split_kwargs: Dict = None,
                 text_splitter: TextSplitter = None) -> None:
        
        # Initialize splitter
        if text_splitter is not None:
            self.doc_splitter = text_splitter
        else:
            split_kwargs = split_kwargs or {"chunk_size": 300, "chunk_overlap": 0}
            self.doc_splitter = TextSplitter(**split_kwargs)
        
        # Initialize document loader
        self.doc_loader = DocumentLoader()

    def load(self, 
             file_paths: Union[str, List[str]], 
             workers: int = None,
             split_documents: bool = True) -> List:
        """
        Load and optionally split documents.
        
        Args:
            file_paths: Single file path or list of file paths
            workers: Number of worker processes (default: number of CPU cores)
            split_documents: Whether to split documents into chunks
            
        Returns:
            List of loaded (and optionally split) documents
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        # Validate all files exist
        valid_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
            else:
                valid_files.append(file_path)
        
        if not valid_files:
            logger.error("No valid files found")
            return []
        
        # Check if all files are the same type
        file_types = set()
        for file_path in valid_files:
            try:
                file_type = validate_file_type(file_path)
                file_types.add(file_type)
            except ValueError:
                continue
        
        # Load documents
        if len(file_types) == 1:
            # All files are the same type
            file_type = next(iter(file_types))
            docs = self.doc_loader.load_single_type(valid_files, file_type, workers)
        else:
            # Mixed file types
            docs = self.doc_loader.load_mixed_types(valid_files, workers)
        
        # Split documents if requested
        if split_documents and docs:
            docs = self.doc_splitter.split_documents(docs)
        
        return docs

    def load_directory(self, 
                      dir_path: str, 
                      workers: int = None,
                      split_documents: bool = True,
                      recursive: bool = False,
                      file_extensions: List[str] = None) -> List:
        """
        Load all supported files from a directory.
        
        Args:
            dir_path: Directory path
            workers: Number of worker processes
            split_documents: Whether to split documents into chunks
            recursive: Whether to search recursively in subdirectories
            file_extensions: List of specific extensions to load (e.g., ['.pdf', '.txt'])
            
        Returns:
            List of loaded (and optionally split) documents
        """
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # Find files
        search_pattern = "**/*.*" if recursive else "*.*"
        all_files = list(Path(dir_path).glob(search_pattern))
        
        # Filter by supported extensions
        valid_files = []
        for file_path in all_files:
            ext = file_path.suffix.lower()
            
            # Check if extension is supported
            if ext in SUPPORTED_EXTENSIONS:
                # Check if specific extensions were requested
                if file_extensions is None or ext in file_extensions:
                    valid_files.append(str(file_path))
        
        if not valid_files:
            logger.warning(f"No supported files found in {dir_path}")
            return []
        
        logger.info(f"Found {len(valid_files)} supported files in {dir_path}")
        return self.load(valid_files, workers, split_documents)

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(SUPPORTED_EXTENSIONS.keys())

    def get_file_info(self, file_paths: Union[str, List[str]]) -> Dict:
        """Get information about files without loading them."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        info = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'files_by_type': {},
            'missing_files': [],
            'invalid_extensions': []
        }
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                info['missing_files'].append(file_path)
                continue
            
            try:
                file_type = validate_file_type(file_path)
                info['valid_files'] += 1
                if file_type not in info['files_by_type']:
                    info['files_by_type'][file_type] = 0
                info['files_by_type'][file_type] += 1
            except ValueError:
                info['invalid_files'] += 1
                ext = get_file_extension(file_path)
                if ext not in info['invalid_extensions']:
                    info['invalid_extensions'].append(ext)
        
        return info


# Example usage and convenience functions
def create_loader(chunk_size: int = 300, chunk_overlap: int = 0) -> Loader:
    """Create a loader with specified chunk settings."""
    return Loader(split_kwargs={
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })


def quick_load(file_path: str, chunk_size: int = 300) -> List:
    """Quickly load a single file with default settings."""
    loader = create_loader(chunk_size=chunk_size)
    return loader.load(file_path)


def load_directory_quick(dir_path: str, chunk_size: int = 300, recursive: bool = False) -> List:
    """Quickly load all files from a directory."""
    loader = create_loader(chunk_size=chunk_size)
    return loader.load_directory(dir_path, recursive=recursive)