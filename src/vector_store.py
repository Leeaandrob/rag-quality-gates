"""
Vector Store Module
===================

This module handles document embedding and vector storage using FAISS.
Supports both OpenAI embeddings and local Nomic embeddings via TEI.

Educational Notes:
- Embeddings convert text to numerical vectors
- Similar texts have similar vector representations
- FAISS enables efficient similarity search at scale
"""

from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class TEIEmbeddings(Embeddings):
    """
    Custom embeddings class for Text Embeddings Inference (TEI) server.

    This allows using local embedding models like Nomic via HuggingFace TEI.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
    ):
        """
        Initialize TEI embeddings client.

        Args:
            base_url: URL of the TEI server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        # Get model info
        try:
            info = self._client.get(f"{self.base_url}/info").json()
            self.model_id = info.get("model_id", "unknown")
            self.max_input_length = info.get("max_input_length", 8192)
            print(f"[TEI] Connected to {self.model_id}")
        except Exception as e:
            print(f"[TEI] Warning: Could not get model info: {e}")
            self.model_id = "unknown"
            self.max_input_length = 8192

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []

        # Process in batches to avoid overloading
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            response = self._client.post(
                f"{self.base_url}/embed",
                json={"inputs": batch},
            )
            response.raise_for_status()
            batch_embeddings = response.json()
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        response = self._client.post(
            f"{self.base_url}/embed",
            json={"inputs": text},
        )
        response.raise_for_status()
        result = response.json()

        # TEI returns list of embeddings even for single input
        return result[0] if isinstance(result[0], list) else result


class VectorStoreManager:
    """
    Manages the vector store for document retrieval.

    Supports:
    - Local embeddings via TEI (Nomic, etc.)
    - OpenAI embeddings (if API key available)

    Key Concepts:
    - Embedding Model: Converts text to vectors
    - Vector Index: Data structure for efficient similarity search
    - Retriever: Interface for querying the vector store
    """

    def __init__(
        self,
        use_local: bool = True,
        tei_url: str = "http://localhost:8080",
        openai_model: str = "text-embedding-3-small",
    ):
        """
        Initialize the vector store manager.

        Args:
            use_local: If True, use local TEI embeddings. If False, use OpenAI.
            tei_url: URL of the TEI server (for local embeddings)
            openai_model: OpenAI embedding model (if not using local)
        """
        self.use_local = use_local

        if use_local:
            self.embeddings = TEIEmbeddings(base_url=tei_url)
            print(f"[VectorStore] Using local TEI embeddings")
        else:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=openai_model)
            print(f"[VectorStore] Using OpenAI embeddings: {openai_model}")

        self.vector_store: Optional[FAISS] = None

    def create_from_documents(self, documents: List[Document]) -> FAISS:
        """
        Create a vector store from documents.

        This is the INGESTION step - documents are:
        1. Embedded (converted to vectors)
        2. Indexed (stored for efficient retrieval)

        Args:
            documents: List of Document objects to index

        Returns:
            FAISS vector store instance
        """
        print(f"[VectorStore] Embedding {len(documents)} documents...")

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )

        print(f"[VectorStore] Index created successfully")
        return self.vector_store

    def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Directory path to save the index
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(path)
        print(f"[VectorStore] Saved to {path}")

    def load(self, path: str) -> FAISS:
        """
        Load a vector store from disk.

        Args:
            path: Directory path containing the saved index

        Returns:
            FAISS vector store instance
        """
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[VectorStore] Loaded from {path}")
        return self.vector_store

    def get_retriever(self, k: int = 4):
        """
        Get a retriever interface for the vector store.

        Args:
            k: Number of documents to retrieve per query

        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search directly.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.

        Useful for debugging retrieval quality.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")

        return self.vector_store.similarity_search_with_score(query, k=k)
