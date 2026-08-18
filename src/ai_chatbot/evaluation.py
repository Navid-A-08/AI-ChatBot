"""
Evaluation framework for measuring chatbot performance.

Why this exists:
    To objectively compare different context strategies (Phase 4)
    and measure the impact of RAG (Phase 3) and memory (Phase 2),
    we need a systematic way to evaluate response quality.

    This framework provides:
    1. Test datasets: Sample queries with expected behaviors
    2. Automated metrics: Response quality, relevance, latency
    3. A/B testing: Compare different configurations
    4. Reporting: Generate evaluation reports

How it works:
    EvaluationDataset contains test cases with queries and expected
    outcomes. Evaluator runs the chatbot on these queries and scores
    the responses. Results are saved to files for analysis.

Alternatives considered:
    - Manual evaluation: Most accurate but doesn't scale.
    - LLM-as-judge: Uses another LLM to score responses (Phase 8).
    - Simple metrics only: Fast but may miss quality issues.

    Current approach provides a baseline that's fast and reproducible,
    with hooks for LLM-based evaluation in Phase 8.

Trade-offs accepted:
    - Simple metrics (length, keyword presence) don't capture response
      quality fully. This is intentional — Phase 8 adds LLM-based
      evaluation for more accurate scoring.
    - Test datasets are small and manually curated. For a portfolio
      project this is fine; production systems would need larger,
      automatically generated datasets.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case."""

    query: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "easy"


@dataclass
class EvalResult:
    """Result of evaluating a single query."""

    query: str
    response: str
    latency_ms: float
    relevance_score: float
    contains_expected: bool
    response_length: int
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregated evaluation results."""

    total_queries: int
    avg_latency_ms: float
    avg_relevance_score: float
    pct_contains_expected: float
    results: list[EvalResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class EvaluationDataset:
    """
    Manages test cases for evaluation.
    """

    def __init__(self, dataset_path: str = None) -> None:
        self._dataset_path = Path(dataset_path) if dataset_path else None
        self._test_cases: list[TestCase] = []

        if self._dataset_path and self._dataset_path.exists():
            self._load_dataset()

    def _load_dataset(self) -> None:
        """Load test cases from a JSON file."""
        try:
            data = json.loads(self._dataset_path.read_text(encoding="utf-8"))
            for item in data:
                self._test_cases.append(TestCase(**item))
            logger.info("Loaded %d test cases", len(self._test_cases))
        except Exception:
            logger.exception("Failed to load evaluation dataset")

    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the dataset."""
        self._test_cases.append(test_case)

    def get_test_cases(self, category: str = None) -> list[TestCase]:
        """
        Get test cases, optionally filtered by category.
        """
        if category:
            return [tc for tc in self._test_cases if tc.category == category]
        return self._test_cases.copy()

    def save_dataset(self, path: str = None) -> None:
        """Save the dataset to a JSON file."""
        save_path = Path(path) if path else self._dataset_path
        if not save_path:
            raise ValueError("No path specified for saving dataset")

        data = [
            {
                "query": tc.query,
                "expected_keywords": tc.expected_keywords,
                "expected_contains": tc.expected_contains,
                "category": tc.category,
                "difficulty": tc.difficulty,
            }
            for tc in self._test_cases
        ]

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved %d test cases to %s", len(data), save_path)


class Evaluator:
    """
    Evaluates chatbot performance on a dataset.
    """

    def __init__(self, chatbot_func) -> None:
        """
        Args:
            chatbot_func: A callable that takes a query string and
                         returns a response string. This should be
                         a function that wraps the chatbot's
                         conversational turn.
        """
        self._chatbot_func = chatbot_func

    def evaluate_query(self, test_case: TestCase) -> EvalResult:
        """
        Evaluate a single query.

        Args:
            test_case: The test case to evaluate.

        Returns:
            EvalResult with metrics for this query.
        """
        start_time = time.time()

        try:
            response = self._chatbot_func(test_case.query)
        except Exception as e:
            logger.exception("Error evaluating query: %s", test_case.query)
            response = f"ERROR: {str(e)}"

        latency_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        relevance_score = self._calculate_relevance(response, test_case)
        contains_expected = self._check_expected(response, test_case)

        return EvalResult(
            query=test_case.query,
            response=response,
            latency_ms=latency_ms,
            relevance_score=relevance_score,
            contains_expected=contains_expected,
            response_length=len(response),
        )

    def evaluate_dataset(
        self, dataset: EvaluationDataset, category: str = None
    ) -> EvalReport:
        """
        Evaluate all queries in a dataset.

        Args:
            dataset: The evaluation dataset.
            category: Optional category filter.

        Returns:
            EvalReport with aggregated metrics.
        """
        test_cases = dataset.get_test_cases(category)
        results = []

        for test_case in test_cases:
            logger.info("Evaluating: %s", test_case.query[:50])
            result = self.evaluate_query(test_case)
            results.append(result)

        # Calculate aggregates
        if not results:
            return EvalReport(
                total_queries=0,
                avg_latency_ms=0,
                avg_relevance_score=0,
                pct_contains_expected=0,
            )

        avg_latency = sum(r.latency_ms for r in results) / len(results)
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        pct_expected = sum(1 for r in results if r.contains_expected) / len(results) * 100

        return EvalReport(
            total_queries=len(results),
            avg_latency_ms=avg_latency,
            avg_relevance_score=avg_relevance,
            pct_contains_expected=pct_expected,
            results=results,
            metadata={"category": category},
        )

    def _calculate_relevance(self, response: str, test_case: TestCase) -> float:
        """
        Calculate relevance score for a response.

        Uses simple keyword matching. Phase 8 will add LLM-based
        evaluation for more accurate scoring.
        """
        # If response is an error, return 0 relevance
        if response.startswith("ERROR:"):
            return 0.0

        if not test_case.expected_keywords:
            return 1.0  # No keywords to check, assume relevant

        response_lower = response.lower()
        keywords_found = sum(
            1 for kw in test_case.expected_keywords
            if kw.lower() in response_lower
        )

        return keywords_found / len(test_case.expected_keywords)

    def _check_expected(self, response: str, test_case: TestCase) -> bool:
        """
        Check if response contains expected substrings.
        """
        if not test_case.expected_contains:
            return True  # Nothing specific expected

        response_lower = response.lower()
        return all(
            expected.lower() in response_lower
            for expected in test_case.expected_contains
        )


def save_report(report: EvalReport, path: str) -> None:
    """Save an evaluation report to a JSON file."""
    data = {
        "total_queries": report.total_queries,
        "avg_latency_ms": report.avg_latency_ms,
        "avg_relevance_score": report.avg_relevance_score,
        "pct_contains_expected": report.pct_contains_expected,
        "metadata": report.metadata,
        "results": [
            {
                "query": r.query,
                "response": r.response[:200] + "..." if len(r.response) > 200 else r.response,
                "latency_ms": r.latency_ms,
                "relevance_score": r.relevance_score,
                "contains_expected": r.contains_expected,
                "response_length": r.response_length,
            }
            for r in report.results
        ],
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Saved evaluation report to %s", path)


def load_default_dataset() -> EvaluationDataset:
    """Load or create a default evaluation dataset."""
    dataset = EvaluationDataset()

    # Basic queries
    dataset.add_test_case(TestCase(
        query="What is Python?",
        expected_keywords=["programming", "language"],
        category="knowledge",
        difficulty="easy",
    ))

    dataset.add_test_case(TestCase(
        query="How do I learn to code?",
        expected_keywords=["learn", "start", "practice"],
        category="advice",
        difficulty="easy",
    ))

    dataset.add_test_case(TestCase(
        query="What's the weather like?",
        expected_keywords=[],
        category="general",
        difficulty="easy",
    ))

    # Memory tests
    dataset.add_test_case(TestCase(
        query="I am a software engineer",
        expected_contains=["understood", "got it", "noted"],
        category="memory",
        difficulty="easy",
    ))

    dataset.add_test_case(TestCase(
        query="What do you know about me?",
        expected_contains=["software engineer"],
        category="memory",
        difficulty="medium",
    ))

    return dataset
