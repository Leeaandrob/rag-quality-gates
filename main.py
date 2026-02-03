#!/usr/bin/env python3
"""
Naive RAG with RAGAS Evaluation
===============================

Educational project demonstrating:
1. PDF document ingestion
2. Vector store creation with local embeddings
3. Naive RAG pipeline with Ollama/Claudex
4. Evaluation with RAGAS metrics

Supports two vector store backends:
- FAISS + TEI (default, requires TEI server)
- Qdrant + SentenceTransformers (Mac Silicon compatible, no server needed)

Usage:
    uv run python main.py

Requirements (FAISS backend - default):
    - Nomic TEI running on localhost:8080
    - Ollama with qwen2.5:3b model OR Claudex

Requirements (Qdrant backend - Mac Silicon):
    - Ollama with qwen2.5:3b model OR Claudex
    - Set VECTOR_BACKEND=qdrant in .env
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from src.document_loader import DocumentProcessor, analyze_chunks
from src.vector_store import VectorStoreManager
from src.rag_pipeline import NaiveRAG
from src.evaluator import RAGASEvaluator, create_test_questions_bitcoin


def main():
    """Main execution pipeline."""

    # Load environment variables
    load_dotenv()

    print("\n" + "=" * 60)
    print("NAIVE RAG WITH RAGAS EVALUATION")
    print("Bitcoin Whitepaper Demo (Local Models)")
    print("=" * 60)

    # Configuration
    PDF_PATH = "bitcoin_paper.pdf"
    OUTPUT_PATH = "outputs/evaluation_results.csv"

    # Vector store backend: "faiss" (default) or "qdrant" (Mac Silicon)
    VECTOR_BACKEND = os.getenv('VECTOR_BACKEND', 'faiss')
    INDEX_PATH = "data/qdrant_metadata" if VECTOR_BACKEND == "qdrant" else "data/faiss_index"

    # Qdrant backend settings
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'bitcoin_docs')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

    # Local infrastructure
    TEI_URL = os.getenv('TEI_URL', 'http://localhost:8080')
    CLAUDEX_URL = os.getenv('CLAUDEX_URL', 'http://localhost:8081/v1')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')
    USE_CLAUDEX = os.getenv('USE_CLAUDEX', 'true').lower() == 'true'

    print(f"\nVector Backend: {VECTOR_BACKEND}")

    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    # =========================================================================
    # STEP 1: Document Ingestion
    # =========================================================================
    print("\n[STEP 1] Document Ingestion")
    print("-" * 40)

    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = processor.process(PDF_PATH)

    # Analyze chunks
    stats = analyze_chunks(chunks)
    print(f"\nChunk Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Show sample chunk
    print(f"\nSample chunk (first 200 chars):")
    print(f"  '{chunks[0].page_content[:200]}...'")

    # =========================================================================
    # STEP 2: Vector Store Creation
    # =========================================================================
    print("\n[STEP 2] Vector Store Creation")
    print("-" * 40)

    if VECTOR_BACKEND == "qdrant":
        vector_manager = VectorStoreManager(
            backend="qdrant",
            collection_name=COLLECTION_NAME,
            embedding_model=EMBEDDING_MODEL,
        )
    else:
        vector_manager = VectorStoreManager(
            backend="faiss",
            use_local=True,
            tei_url=TEI_URL,
        )

    # Create index from chunks
    vector_manager.create_from_documents(chunks)

    # Save for later use
    vector_manager.save(INDEX_PATH)

    # Test retrieval
    test_query = "What is Bitcoin?"
    print(f"\nTest retrieval for: '{test_query}'")
    results = vector_manager.similarity_search_with_score(test_query, k=2)
    for doc, score in results:
        print(f"  Score: {score:.4f} | Page: {doc.metadata.get('page', '?')}")
        print(f"  Content: {doc.page_content[:100]}...")

    # =========================================================================
    # STEP 3: RAG Pipeline Setup
    # =========================================================================
    print("\n[STEP 3] RAG Pipeline Setup")
    print("-" * 40)

    rag = NaiveRAG(
        vector_store_manager=vector_manager,
        use_local_llm=not USE_CLAUDEX,
        use_claudex=USE_CLAUDEX,
        claudex_url=CLAUDEX_URL,
        ollama_model=OLLAMA_MODEL,
        k=4,
    )

    # Test single query
    print("\nTest query: 'What is Bitcoin?'")
    result = rag.query("What is Bitcoin?")
    print(f"Response: {result['response'][:300]}...")

    # =========================================================================
    # STEP 4: Generate Test Dataset
    # =========================================================================
    print("\n[STEP 4] Generate Evaluation Dataset")
    print("-" * 40)

    questions, references = create_test_questions_bitcoin()
    print(f"Created {len(questions)} test questions")

    # Run RAG on all questions
    print("Processing questions through RAG pipeline...")
    results = rag.batch_query(questions, references)

    # =========================================================================
    # STEP 5: RAGAS Evaluation
    # =========================================================================
    print("\n[STEP 5] RAGAS Evaluation")
    print("-" * 40)

    # Initialize evaluator
    evaluator = RAGASEvaluator(
        metrics=["faithfulness", "answer_relevancy"],
        use_local=not USE_CLAUDEX,
        use_claudex=USE_CLAUDEX,
        claudex_url=CLAUDEX_URL,
        ollama_model=OLLAMA_MODEL,
        tei_url=TEI_URL,
        vector_backend=VECTOR_BACKEND,
    )

    # Run evaluation
    evaluation = evaluator.evaluate(results)

    # Print report
    evaluator.print_report(evaluation)

    # Save detailed results
    evaluator.save_results(evaluation, OUTPUT_PATH)

    # =========================================================================
    # STEP 6: Analysis & Insights
    # =========================================================================
    print("\n[STEP 6] Analysis & Insights")
    print("-" * 40)

    df = evaluation["dataframe"]

    print("\nPer-question scores:")
    for _, row in df.iterrows():
        q = row.get("user_input", "")[:50]
        faith = row.get("faithfulness", 0)
        rel = row.get("answer_relevancy", 0)
        print(f"  Q: {q}...")
        print(f"    Faithfulness: {faith:.2f} | Relevancy: {rel:.2f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"\nInfrastructure Used (Backend: {VECTOR_BACKEND}):")
    if VECTOR_BACKEND == "qdrant":
        print(f"  - Embeddings: {EMBEDDING_MODEL} (local CPU)")
        print(f"  - Vector Store: Qdrant (in-memory)")
    else:
        print(f"  - Embeddings: Nomic via TEI ({TEI_URL})")
        print(f"  - Vector Store: FAISS (local)")
    if USE_CLAUDEX:
        print(f"  - LLM: Claudex ({CLAUDEX_URL})")
    else:
        print(f"  - LLM: Ollama ({OLLAMA_MODEL})")
    print("\nNext steps:")
    print("  1. Review low-scoring questions")
    print("  2. Experiment with chunk_size and chunk_overlap")
    print("  3. Try different Ollama models")
    print("  4. Adjust retrieval k parameter")


if __name__ == "__main__":
    main()
