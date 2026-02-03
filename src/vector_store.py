"""
Vector Store Module
===================

This module handles document embedding and vector storage.
Supports two backends:
- FAISS with TEI embeddings (default, requires TEI server)
- Qdrant with SentenceTransformers (Mac Silicon compatible, no server needed)

Select backend via VECTOR_BACKEND environment variable.

Educational Notes:
- Embeddings convert text to numerical vectors
- Similar texts have similar vector representations
- Both FAISS and Qdrant enable efficient similarity search at scale
"""

from pathlib import Path
from typing import List, Optional

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
        import httpx

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
        import httpx

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

    Supports two backends:
    - "faiss": FAISS + TEI embeddings (default, requires TEI server)
    - "qdrant": Qdrant in-memory + SentenceTransformers (Mac Silicon compatible)

    Key Concepts:
    - Embedding Model: Converts text to vectors
    - Vector Index: Data structure for efficient similarity search
    - Retriever: Interface for querying the vector store
    """

    def __init__(
        self,
        backend: str = "faiss",
        # FAISS backend params (original)
        use_local: bool = True,
        tei_url: str = "http://localhost:8080",
        openai_model: str = "text-embedding-3-small",
        # Qdrant backend params
        collection_name: str = "bitcoin_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        """
        Initialize the vector store manager.

        Args:
            backend: "faiss" (default) or "qdrant" (Mac Silicon compatible)
            use_local: If True, use local TEI embeddings. If False, use OpenAI. (FAISS only)
            tei_url: URL of the TEI server (FAISS only)
            openai_model: OpenAI embedding model (FAISS only)
            collection_name: Name of the Qdrant collection (Qdrant only)
            embedding_model: HuggingFace model for local embeddings (Qdrant only)
            qdrant_url: URL of remote Qdrant instance (Qdrant only, optional)
            qdrant_api_key: API key for remote Qdrant (Qdrant only, optional)
        """
        self.backend = backend
        self.vector_store = None

        if backend == "qdrant":
            self._init_qdrant(collection_name, embedding_model, qdrant_url, qdrant_api_key)
        else:
            self._init_faiss(use_local, tei_url, openai_model)

    def _init_faiss(self, use_local: bool, tei_url: str, openai_model: str):
        """Initialize FAISS backend with TEI or OpenAI embeddings."""
        self.use_local = use_local

        if use_local:
            self.embeddings = TEIEmbeddings(base_url=tei_url)
            print(f"[VectorStore] Using FAISS with local TEI embeddings")
        else:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=openai_model)
            print(f"[VectorStore] Using FAISS with OpenAI embeddings: {openai_model}")

    def _init_qdrant(
        self,
        collection_name: str,
        embedding_model: str,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
    ):
        """Initialize Qdrant backend with local SentenceTransformers embeddings."""
        self.collection_name = collection_name

        self.embeddings = get_local_embeddings(embedding_model)
        print(f"[VectorStore] Using Qdrant with local embeddings: {embedding_model}")

        from qdrant_client import QdrantClient

        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print(f"[VectorStore] Connected to remote Qdrant: {qdrant_url}")
        else:
            self.qdrant_client = QdrantClient(":memory:")
            print(f"[VectorStore] Using in-memory Qdrant (no server needed)")

    def create_from_documents(self, documents: List[Document]):
        """
        Create a vector store from documents.

        This is the INGESTION step - documents are:
        1. Embedded (converted to vectors)
        2. Indexed (stored for efficient retrieval)

        Args:
            documents: List of Document objects to index

        Returns:
            Vector store instance
        """
        print(f"[VectorStore] Embedding {len(documents)} documents...")

        if self.backend == "qdrant":
            from langchain_qdrant import QdrantVectorStore

            self.vector_store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                location=":memory:",
            )
            print(f"[VectorStore] Qdrant collection '{self.collection_name}' created")
        else:
            from langchain_community.vectorstores import FAISS

            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )
            print(f"[VectorStore] FAISS index created successfully")

        return self.vector_store

    def save(self, path: str) -> None:
        """
        Save the vector store to disk.

        Args:
            path: Directory path to save the index
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        if self.backend == "qdrant":
            import json

            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)
            metadata = {
                "collection_name": self.collection_name,
                "embedding_model": self.embeddings.model_name,
                "backend": "qdrant",
            }
            with open(save_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[VectorStore] Metadata saved to {path}")
            print(f"[VectorStore] Note: In-memory data persists only during session")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(path)
            print(f"[VectorStore] Saved to {path}")

    def load(self, path: str):
        """
        Load a vector store from disk.

        Args:
            path: Directory path containing the saved index

        Returns:
            Vector store instance
        """
        if self.backend == "qdrant":
            import json

            with open(f"{path}/metadata.json", "r") as f:
                metadata = json.load(f)
            print(f"[VectorStore] Loaded metadata from {path}")
            print(f"[VectorStore] Collection: {metadata['collection_name']}")

            if self.vector_store is None:
                raise ValueError("Collection not created. Run create_from_documents first.")
            return self.vector_store
        else:
            from langchain_community.vectorstores import FAISS

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


def get_local_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get local SentenceTransformers embeddings (Mac Silicon compatible).

    Used by the Qdrant backend and evaluator module.

    Args:
        model_name: HuggingFace model name

    Returns:
        HuggingFaceEmbeddings instance
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
