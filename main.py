#!/usr/bin/env python3
"""
Naive RAG with RAGAS Evaluation
===============================

Educational project demonstrating:
1. PDF document ingestion
2. Vector store creation (FAISS) with local Nomic embeddings
3. Naive RAG pipeline with Ollama
4. Evaluation with RAGAS metrics

Usage:
    uv run python main.py

Requirements:
    - Nomic TEI running on localhost:8080
    - Ollama with qwen2.5:3b model
"""

import os
import sys
from pathlib import Path

from src.document_loader import DocumentProcessor, analyze_chunks
from src.vector_store import VectorStoreManager
from src.rag_pipeline import NaiveRAG
from src.evaluator import RAGASEvaluator, create_test_questions_bitcoin


def main():
    """Main execution pipeline."""

    print("\n" + "=" * 60)
    print("NAIVE RAG WITH RAGAS EVALUATION")
    print("Bitcoin Whitepaper Demo (Local Models)")
    print("=" * 60)

    # Configuration
    PDF_PATH = "bitcoin_paper.pdf"
    INDEX_PATH = "data/faiss_index"
    OUTPUT_PATH = "outputs/evaluation_results.csv"

    # Local infrastructure
    TEI_URL = "http://localhost:8080"
    CLAUDEX_URL = "http://localhost:8081/v1"
    OLLAMA_MODEL = "qwen2.5:3b"  # Fallback if Claudex is unavailable
    USE_CLAUDEX = True  # Set to False to use Ollama instead

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
    # STEP 2: Vector Store Creation (Local Nomic Embeddings)
    # =========================================================================
    print("\n[STEP 2] Vector Store Creation")
    print("-" * 40)

    vector_manager = VectorStoreManager(
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
    # STEP 3: RAG Pipeline Setup (Local Ollama)
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
    # STEP 5: RAGAS Evaluation (Local Models)
    # =========================================================================
    print("\n[STEP 5] RAGAS Evaluation")
    print("-" * 40)

    # Initialize evaluator with local models
    evaluator = RAGASEvaluator(
        metrics=["faithfulness", "answer_relevancy"],
        use_local=not USE_CLAUDEX,
        use_claudex=USE_CLAUDEX,
        claudex_url=CLAUDEX_URL,
        ollama_model=OLLAMA_MODEL,
        tei_url=TEI_URL,
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
    print("\nLocal Infrastructure Used:")
    print(f"  - Embeddings: Nomic via TEI ({TEI_URL})")
    if USE_CLAUDEX:
        print(f"  - LLM: Claudex ({CLAUDEX_URL})")
    else:
        print(f"  - LLM: Ollama ({OLLAMA_MODEL})")
    print(f"  - Vector Store: FAISS (local)")
    print("\nNext steps:")
    print("  1. Review low-scoring questions")
    print("  2. Experiment with chunk_size and chunk_overlap")
    print("  3. Try different Ollama models")
    print("  4. Adjust retrieval k parameter")


if __name__ == "__main__":
    main()
