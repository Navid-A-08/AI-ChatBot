from ai_chatbot.evaluation import (
    TestCase,
    EvalResult,
    EvalReport,
    EvaluationDataset,
    Evaluator,
    load_default_dataset,
)


def test_test_case_creation():
    """Test creating a TestCase."""
    tc = TestCase(
        query="What is Python?",
        expected_keywords=["programming", "language"],
        category="knowledge",
    )

    assert tc.query == "What is Python?"
    assert "programming" in tc.expected_keywords
    assert tc.category == "knowledge"


def test_evaluation_dataset():
    """Test EvaluationDataset operations."""
    dataset = EvaluationDataset()

    dataset.add_test_case(TestCase(query="Query 1"))
    dataset.add_test_case(TestCase(query="Query 2", category="special"))
    dataset.add_test_case(TestCase(query="Query 3", category="special"))

    # Get all
    all_cases = dataset.get_test_cases()
    assert len(all_cases) == 3

    # Get by category
    special_cases = dataset.get_test_cases(category="special")
    assert len(special_cases) == 2


def test_eval_result_creation():
    """Test creating an EvalResult."""
    result = EvalResult(
        query="Test query",
        response="Test response",
        latency_ms=150.5,
        relevance_score=0.8,
        contains_expected=True,
        response_length=13,
    )

    assert result.query == "Test query"
    assert result.latency_ms == 150.5
    assert result.relevance_score == 0.8


def test_eval_report_creation():
    """Test creating an EvalReport."""
    report = EvalReport(
        total_queries=10,
        avg_latency_ms=200.0,
        avg_relevance_score=0.75,
        pct_contains_expected=80.0,
    )

    assert report.total_queries == 10
    assert report.avg_latency_ms == 200.0


def test_evaluator_basic():
    """Test Evaluator with a simple chatbot function."""
    def mock_chatbot(query: str) -> str:
        return f"I can help with: {query}"

    evaluator = Evaluator(mock_chatbot)
    result = evaluator.evaluate_query(TestCase(query="Python"))

    assert "Python" in result.response
    assert result.latency_ms >= 0


def test_evaluator_relevance_scoring():
    """Test relevance scoring in Evaluator."""
    def mock_chatbot(query: str) -> str:
        return "Python is a programming language for software development."

    evaluator = Evaluator(mock_chatbot)
    result = evaluator.evaluate_query(TestCase(
        query="Tell me about Python",
        expected_keywords=["programming", "language"],
    ))

    assert result.relevance_score > 0.5  # Should find both keywords


def test_evaluator_expected_contains():
    """Test expected_contains checking."""
    def mock_chatbot(query: str) -> str:
        return "Got it! I understand."

    evaluator = Evaluator(mock_chatbot)

    # Should pass
    result = evaluator.evaluate_query(TestCase(
        query="Remember this",
        expected_contains=["got it", "understand"],
    ))
    assert result.contains_expected is True

    # Should fail
    result = evaluator.evaluate_query(TestCase(
        query="Remember this",
        expected_contains=["memory stored"],
    ))
    assert result.contains_expected is False


def test_evaluator_evaluate_dataset():
    """Test evaluating a full dataset."""
    def mock_chatbot(query: str) -> str:
        return f"Response to: {query}"

    dataset = EvaluationDataset()
    dataset.add_test_case(TestCase(query="Q1"))
    dataset.add_test_case(TestCase(query="Q2"))

    evaluator = Evaluator(mock_chatbot)
    report = evaluator.evaluate_dataset(dataset)

    assert report.total_queries == 2
    assert report.avg_latency_ms >= 0


def test_load_default_dataset():
    """Test loading the default dataset."""
    dataset = load_default_dataset()

    test_cases = dataset.get_test_cases()
    assert len(test_cases) >= 3

    # Check categories exist
    categories = set(tc.category for tc in test_cases)
    assert "knowledge" in categories
    assert "memory" in categories


def test_evaluator_handles_errors():
    """Test that evaluator handles chatbot errors gracefully."""
    def failing_chatbot(query: str) -> str:
        raise ValueError("Chatbot crashed")

    evaluator = Evaluator(failing_chatbot)
    result = evaluator.evaluate_query(TestCase(query="Test"))

    assert "ERROR" in result.response
    assert result.relevance_score == 0.0
