"""
RAG Pipeline Module
===================

This module implements a naive RAG (Retrieval-Augmented Generation) pipeline.
Supports OpenAI, local Ollama models, and Claudex (OpenAI-compatible Claude wrapper).

Educational Notes:
- RAG = Retrieval + Generation
- Retrieved context is used to ground LLM responses
- Quality depends on both retrieval and generation quality
"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

from .vector_store import VectorStoreManager


class NaiveRAG:
    """
    A simple RAG implementation for educational purposes.

    Supports:
    - Local LLMs via Ollama
    - OpenAI models (if API key available)
    - Claudex (OpenAI-compatible Claude wrapper)

    Pipeline:
    1. User asks a question
    2. Retrieve relevant documents from vector store
    3. Construct prompt with question + context
    4. Generate answer using LLM
    """

    # System prompt for the RAG assistant
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use information from the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Quote relevant parts when helpful"""

    # User prompt template
    USER_PROMPT = """Context:
{context}

Question: {question}

Answer based on the context above:"""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        use_local_llm: bool = True,
        use_claudex: bool = False,
        claudex_url: str = "http://localhost:8081/v1",
        ollama_model: str = "qwen2.5:3b",
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        k: int = 4,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store_manager: Initialized vector store manager
            use_local_llm: If True, use Ollama. If False, use OpenAI/Claudex.
            use_claudex: If True, use Claudex (OpenAI-compatible Claude wrapper).
            claudex_url: URL of the Claudex server.
            ollama_model: Ollama model to use
            openai_model: OpenAI model to use
            temperature: LLM temperature (0 = deterministic)
            k: Number of documents to retrieve
        """
        self.vector_store = vector_store_manager
        self.k = k
        self.use_local_llm = use_local_llm
        self.use_claudex = use_claudex

        # Initialize LLM
        if use_claudex:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model="claude",
                base_url=claudex_url,
                api_key="not-needed",  # Auth handled by Claudex
                temperature=temperature,
            )
            print(f"[RAG] Using Claudex at: {claudex_url}")
        elif use_local_llm:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=ollama_model,
                temperature=temperature,
            )
            print(f"[RAG] Using Ollama model: {ollama_model}")
        else:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
            )
            print(f"[RAG] Using OpenAI model: {openai_model}")

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.USER_PROMPT),
        ])

        print(f"[RAG] Initialized with k={k}")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User question

        Returns:
            List of relevant documents
        """
        documents = self.vector_store.similarity_search(query, k=self.k)
        return documents

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("page", "unknown")
            context_parts.append(f"[Source: Page {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Generated answer
        """
        messages = self.prompt.format_messages(
            context=context,
            question=query,
        )

        response = self.llm.invoke(messages)
        return response.content

    def query(self, question: str) -> Dict[str, Any]:
        """
        Full RAG pipeline: retrieve + generate.

        Args:
            question: User question

        Returns:
            Dictionary with question, contexts, and response
        """
        # Step 1: Retrieve
        documents = self.retrieve(question)
        contexts = [doc.page_content for doc in documents]

        # Step 2: Format context
        formatted_context = self.format_context(documents)

        # Step 3: Generate
        response = self.generate(question, formatted_context)

        return {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": response,
        }

    def batch_query(
        self,
        questions: List[str],
        references: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions (for evaluation).

        Args:
            questions: List of questions
            references: Optional list of ground truth answers

        Returns:
            List of result dictionaries
        """
        results = []

        for i, question in enumerate(questions):
            print(f"[RAG] Processing question {i+1}/{len(questions)}")
            result = self.query(question)

            # Add reference if provided
            if references and i < len(references):
                result["reference"] = references[i]

            results.append(result)

        return results
