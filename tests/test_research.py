import tempfile
from pathlib import Path

from ai_chatbot.research import ExperimentConfig, ExperimentResult, ExperimentRunner
from ai_chatbot.evaluation import EvaluationDataset, TestCase


def test_experiment_config_creation():
    """Test creating an ExperimentConfig."""
    config = ExperimentConfig(
        name="test_experiment",
        description="A test experiment",
        use_memory=True,
        use_rag=False,
        max_history_turns=4,
    )

    assert config.name == "test_experiment"
    assert config.use_memory is True
    assert config.use_rag is False
    assert config.max_history_turns == 4


def test_experiment_result_creation():
    """Test creating an ExperimentResult."""
    from ai_chatbot.evaluation import EvalReport

    config = ExperimentConfig(name="test")
    report = EvalReport(
        total_queries=10,
        avg_latency_ms=100.0,
        avg_relevance_score=0.8,
        pct_contains_expected=75.0,
    )

    result = ExperimentResult(config=config, eval_report=report)

    assert result.config.name == "test"
    assert result.eval_report.total_queries == 10


def test_experiment_runner_initialization():
    """Test ExperimentRunner initialization."""
    def mock_chatbot(query: str, config: ExperimentConfig) -> str:
        return f"Response to: {query}"

    runner = ExperimentRunner(mock_chatbot)
    assert runner is not None


def test_run_experiment():
    """Test running a single experiment."""
    def mock_chatbot(query: str, config: ExperimentConfig) -> str:
        return f"Response to: {query}"

    dataset = EvaluationDataset()
    dataset.add_test_case(TestCase(query="Test query"))

    runner = ExperimentRunner(mock_chatbot)
    config = ExperimentConfig(name="test_run")

    result = runner.run_experiment(config, dataset)

    assert result.config.name == "test_run"
    assert result.eval_report.total_queries == 1


def test_run_experiment_saves_result():
    """Test that running an experiment saves results to file."""
    def mock_chatbot(query: str, config: ExperimentConfig) -> str:
        return "Test response"

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(mock_chatbot)
        runner._results_dir = Path(tmpdir)

        dataset = EvaluationDataset()
        dataset.add_test_case(TestCase(query="Test"))

        config = ExperimentConfig(name="saved_experiment")
        runner.run_experiment(config, dataset)

        # Check that result file was created
        result_file = Path(tmpdir) / "saved_experiment.json"
        assert result_file.exists()


def test_run_ablation_study():
    """Test running an ablation study."""
    def mock_chatbot(query: str, config: ExperimentConfig) -> str:
        return f"Response with memory={config.use_memory}, rag={config.use_rag}"

    dataset = EvaluationDataset()
    dataset.add_test_case(TestCase(query="Test query"))

    runner = ExperimentRunner(mock_chatbot)
    results = runner.run_ablation_study(dataset)

    # Should have 5 experiments (base, no_memory, no_rag, minimal, ranked)
    assert len(results) == 5

    # Check that different configs were tested
    names = [r.config.name for r in results]
    assert "base" in names
    assert "no_memory" in names
    assert "no_rag" in names


def test_compare_results():
    """Test comparing experiment results."""
    def mock_chatbot(query: str, config: ExperimentConfig) -> str:
        return "Response"

    dataset = EvaluationDataset()
    dataset.add_test_case(TestCase(query="Test"))

    runner = ExperimentRunner(mock_chatbot)

    # Run two experiments
    result1 = runner.run_experiment(
        ExperimentConfig(name="fast", description="Fast config"),
        dataset,
    )
    result2 = runner.run_experiment(
        ExperimentConfig(name="accurate", description="Accurate config"),
        dataset,
    )

    comparison = runner.compare_results([result1, result2])

    assert "experiments" in comparison
    assert len(comparison["experiments"]) == 2
    assert "best_latency" in comparison


def test_experiment_config_with_tags():
    """Test ExperimentConfig with tags."""
    config = ExperimentConfig(
        name="tagged_experiment",
        tags=["ablation", "memory"],
    )

    assert "ablation" in config.tags
    assert "memory" in config.tags
