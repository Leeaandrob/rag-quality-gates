"""
RAGAS Evaluator Module
======================

This module handles RAG evaluation using the RAGAS framework.
Supports both OpenAI and local Ollama models for evaluation.

Educational Notes:
- RAGAS provides reference-free evaluation metrics
- Metrics cover both retrieval and generation quality
- LLM-as-a-judge approach for automated evaluation
"""

from typing import List, Dict, Any, Optional

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    FactualCorrectness,
    SemanticSimilarity,
)


class RAGASEvaluator:
    """
    Evaluates RAG pipeline performance using RAGAS metrics.

    Supports:
    - Local evaluation with Ollama
    - OpenAI-based evaluation

    Available Metrics:
    - faithfulness: Is the answer grounded in the context?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: Are relevant chunks ranked higher?
    - context_recall: Was all needed information retrieved?
    """

    # Metric descriptions for educational purposes
    METRIC_DESCRIPTIONS = {
        "faithfulness": (
            "Measures factual accuracy of the answer based on context. "
            "High = answer is grounded in retrieved documents. "
            "Low = answer contains hallucinations."
        ),
        "answer_relevancy": (
            "Measures how relevant the answer is to the question. "
            "High = answer directly addresses the question. "
            "Low = answer is off-topic or incomplete."
        ),
        "context_precision": (
            "Measures if relevant chunks are ranked at the top. "
            "High = retrieval is precise. "
            "Low = irrelevant chunks are mixed in."
        ),
        "context_recall": (
            "Measures if all needed information was retrieved. "
            "High = complete information retrieval. "
            "Low = missing important context. "
            "NOTE: Requires ground truth reference."
        ),
        "factual_correctness": (
            "Measures factual accuracy against reference. "
            "High = facts match reference. "
            "NOTE: Requires ground truth reference."
        ),
        "semantic_similarity": (
            "Semantic similarity between answer and reference. "
            "High = answer matches expected response. "
            "NOTE: Requires ground truth reference."
        ),
    }

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        use_local: bool = True,
        use_claudex: bool = False,
        claudex_url: str = "http://localhost:8081/v1",
        ollama_model: str = "qwen2.5:3b",
        tei_url: str = "http://localhost:8080",
        vector_backend: str = "faiss",
    ):
        """
        Initialize the evaluator with specified metrics.

        Args:
            metrics: List of metric names to use.
            use_local: If True, use local Ollama for evaluation.
            use_claudex: If True, use Claudex for evaluation.
            claudex_url: URL of Claudex server.
            ollama_model: Ollama model to use for evaluation.
            tei_url: URL of TEI server for embeddings (FAISS backend only).
            vector_backend: "faiss" or "qdrant". Determines embeddings source.
        """
        self.use_local = use_local
        self.use_claudex = use_claudex
        self.claudex_url = claudex_url
        self.ollama_model = ollama_model
        self.tei_url = tei_url
        self.vector_backend = vector_backend

        # Default metrics (no ground truth required)
        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy"]

        self.metric_names = metrics
        self._setup_metrics()

        print(f"[Evaluator] Initialized with metrics: {metrics}")
        if use_claudex:
            print(f"[Evaluator] Using Claudex at {claudex_url} for evaluation")
        elif use_local:
            print(f"[Evaluator] Using local Ollama for evaluation")
        else:
            print(f"[Evaluator] Using OpenAI for evaluation")

    def _get_embeddings(self):
        """Get embeddings based on vector backend configuration."""
        if self.vector_backend == "qdrant":
            from src.vector_store import get_local_embeddings
            return get_local_embeddings()
        else:
            from src.vector_store import TEIEmbeddings
            return TEIEmbeddings(base_url=self.tei_url)

    def _setup_metrics(self):
        """Setup RAGAS metrics with appropriate LLM/embeddings."""
        if self.use_claudex:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm = ChatOpenAI(
                model="claude",
                base_url=self.claudex_url,
                api_key="not-needed",
                temperature=0,
            )
            embeddings = self._get_embeddings()

            wrapped_llm = LangchainLLMWrapper(llm)
            wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

            # Create metrics with Claudex
            self.metrics = []
            for name in self.metric_names:
                if name == "faithfulness":
                    self.metrics.append(Faithfulness(llm=wrapped_llm))
                elif name == "answer_relevancy":
                    self.metrics.append(ResponseRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings))
                elif name == "context_precision":
                    self.metrics.append(LLMContextPrecisionWithReference(llm=wrapped_llm))
                elif name == "context_recall":
                    self.metrics.append(LLMContextRecall(llm=wrapped_llm))
                elif name == "factual_correctness":
                    self.metrics.append(FactualCorrectness(llm=wrapped_llm))
                elif name == "semantic_similarity":
                    self.metrics.append(SemanticSimilarity(embeddings=wrapped_embeddings))
        elif self.use_local:
            from langchain_ollama import ChatOllama
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm = ChatOllama(model=self.ollama_model, temperature=0)
            embeddings = self._get_embeddings()

            wrapped_llm = LangchainLLMWrapper(llm)
            wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

            # Create metrics with local models
            self.metrics = []
            for name in self.metric_names:
                if name == "faithfulness":
                    self.metrics.append(Faithfulness(llm=wrapped_llm))
                elif name == "answer_relevancy":
                    self.metrics.append(ResponseRelevancy(llm=wrapped_llm, embeddings=wrapped_embeddings))
                elif name == "context_precision":
                    self.metrics.append(LLMContextPrecisionWithReference(llm=wrapped_llm))
                elif name == "context_recall":
                    self.metrics.append(LLMContextRecall(llm=wrapped_llm))
                elif name == "factual_correctness":
                    self.metrics.append(FactualCorrectness(llm=wrapped_llm))
                elif name == "semantic_similarity":
                    self.metrics.append(SemanticSimilarity(embeddings=wrapped_embeddings))
        else:
            # Use default OpenAI-based metrics
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            self.metrics = [metric_map[m] for m in self.metric_names if m in metric_map]

    def prepare_dataset(
        self,
        results: List[Dict[str, Any]],
    ) -> Dataset:
        """
        Convert RAG results to RAGAS-compatible dataset.

        Required fields:
        - user_input: The question
        - response: Generated answer
        - retrieved_contexts: List of context strings

        Optional fields:
        - reference: Ground truth answer (for some metrics)

        Args:
            results: List of RAG result dictionaries

        Returns:
            HuggingFace Dataset
        """
        # Validate required fields
        required = ["user_input", "response", "retrieved_contexts"]
        for result in results:
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

        return Dataset.from_list(results)

    def evaluate(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate RAG results using RAGAS metrics.

        Args:
            results: List of RAG result dictionaries

        Returns:
            Dictionary with evaluation scores and details
        """
        print(f"[Evaluator] Evaluating {len(results)} samples...")

        # Prepare dataset
        dataset = self.prepare_dataset(results)

        # Run evaluation
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )

        # Convert to pandas DataFrame
        df = evaluation_result.to_pandas()

        # Extract scores from DataFrame columns
        scores = {}
        for metric in self.metrics:
            metric_name = getattr(metric, 'name', str(metric))
            if metric_name in df.columns:
                # Calculate mean score across all samples
                scores[metric_name] = float(df[metric_name].mean())

        print(f"[Evaluator] Evaluation complete!")
        return {
            "scores": scores,
            "dataframe": df,
        }

    def print_report(
        self,
        evaluation_result: Dict[str, Any],
    ) -> None:
        """
        Print a formatted evaluation report.

        Args:
            evaluation_result: Result from evaluate()
        """
        scores = evaluation_result["scores"]

        print("\n" + "=" * 60)
        print("RAGAS EVALUATION REPORT")
        print("=" * 60)

        for metric, score in scores.items():
            # Color-code based on score
            if score >= 0.8:
                status = "GOOD"
            elif score >= 0.6:
                status = "FAIR"
            else:
                status = "NEEDS IMPROVEMENT"

            print(f"\n{metric.upper()}: {score:.3f} [{status}]")
            desc = self.METRIC_DESCRIPTIONS.get(metric, "")
            if desc:
                print(f"  {desc[:80]}...")

        print("\n" + "=" * 60)

        # Calculate average
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            print(f"AVERAGE SCORE: {avg_score:.3f}")
        print("=" * 60)

    def save_results(
        self,
        evaluation_result: Dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Save evaluation results to CSV.

        Args:
            evaluation_result: Result from evaluate()
            output_path: Path for output CSV file
        """
        df = evaluation_result["dataframe"]
        df.to_csv(output_path, index=False)
        print(f"[Evaluator] Results saved to {output_path}")


def create_test_questions_bitcoin() -> tuple:
    """
    Create test questions about the Bitcoin whitepaper.

    Returns:
        Tuple of (questions, references)
    """
    questions = [
        "What is Bitcoin according to the whitepaper?",
        "How does Bitcoin prevent double-spending?",
        "What is the role of proof-of-work in Bitcoin?",
        "How are transactions verified in Bitcoin?",
        "What is a blockchain in the context of Bitcoin?",
    ]

    # Ground truth references (used for context_recall, answer_similarity)
    references = [
        "Bitcoin is a peer-to-peer electronic cash system that allows online payments to be sent directly from one party to another without going through a financial institution.",
        "Bitcoin prevents double-spending through a peer-to-peer network using proof-of-work to record a public history of transactions, making it computationally impractical for an attacker to change.",
        "Proof-of-work involves scanning for a value that when hashed with SHA-256, the hash begins with a number of zero bits. It secures the network by requiring computational effort to create blocks.",
        "Transactions are verified by network nodes through cryptography and recorded in a public distributed ledger called a blockchain.",
        "The blockchain is a chain of blocks containing transaction data, where each block includes a hash of the previous block, creating a linked chain of records.",
    ]

    return questions, references
