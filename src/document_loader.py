"""
Document Loader Module
======================

This module handles PDF document loading and text chunking.

Educational Notes:
- Chunk size affects retrieval granularity
- Chunk overlap helps maintain context across boundaries
- Metadata preservation is important for traceability
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """
    Processes PDF documents into chunks suitable for RAG.

    The chunking strategy significantly impacts RAG performance:
    - Too small: loses context, retrieves incomplete information
    - Too large: includes irrelevant information, wastes tokens
    - Overlap: helps maintain context at chunk boundaries
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return raw documents (one per page).

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects (one per page)
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        loader = PyPDFLoader(str(path))
        documents = loader.load()

        print(f"[Loader] Loaded {len(documents)} pages from {path.name}")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk index to metadata for traceability
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print(f"[Splitter] Created {len(chunks)} chunks")
        print(f"[Splitter] Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

        return chunks

    def process(self, pdf_path: str) -> List[Document]:
        """
        Full pipeline: load PDF and split into chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of chunked Document objects
        """
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        return chunks


def analyze_chunks(chunks: List[Document]) -> dict:
    """
    Analyze chunk statistics for debugging/optimization.

    Args:
        chunks: List of Document objects

    Returns:
        Dictionary with chunk statistics
    """
    sizes = [len(c.page_content) for c in chunks]

    stats = {
        "total_chunks": len(chunks),
        "total_characters": sum(sizes),
        "avg_chunk_size": sum(sizes) // len(sizes) if sizes else 0,
        "min_chunk_size": min(sizes) if sizes else 0,
        "max_chunk_size": max(sizes) if sizes else 0,
        "pages_covered": len(set(c.metadata.get("page", 0) for c in chunks)),
    }

    return stats
