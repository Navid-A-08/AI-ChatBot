"""
Research comparison framework for ablation studies.

Why this exists:
    The whole point of this project is to compare different context
    engineering strategies. This module provides the infrastructure
    for running controlled experiments:

    1. A/B testing: Compare two configurations side-by-side
    2. Ablation studies: Systematically remove components to measure
       their impact (e.g., with/without memory, with/without RAG)
    3. Parameter sweeps: Test different values for hyperparameters
       (window size, top-k, etc.)

How it works:
    ExperimentRunner takes a configuration, runs it on a test dataset,
    and collects metrics. Results are saved for analysis and comparison.

    Each experiment tests one variable while keeping others constant:
    - Context strategy (window vs ranked)
    - Memory enabled/disabled
    - RAG enabled/disabled
    - Tool use enabled/disabled
    - Window size (2, 4, 6, 8, 10)
    - Top-k for RAG (1, 3, 5, 7)

Alternatives considered:
    - Manual testing: Not systematic, hard to reproduce.
    - External experiment frameworks (MLflow, etc.): Overkill for
      a portfolio project. Simple file-based results are sufficient.
    - Statistical significance testing: Not needed for small-scale
      comparisons, but the framework supports adding it later.

Trade-offs accepted:
    - Results are stored as JSON files, not in a database. Fine for
      a portfolio project.
    - Limited statistical analysis. Phase 8 focuses on collecting
      metrics; interpretation is manual.
    - Experiments are run sequentially, not in parallel. Could be
      parallelized for larger studies.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ai_chatbot.context import ConversationHistory
from ai_chatbot.context_ranker import ContextRanker
from ai_chatbot.evaluation import EvalReport, EvaluationDataset, Evaluator
from ai_chatbot.memory import MemoryManager
from ai_chatbot.rag import RAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str = ""

    # Component toggles
    use_memory: bool = True
    use_rag: bool = True
    use_tools: bool = False
    use_ranked_context: bool = False

    # Parameters
    max_history_turns: int = 6
    max_context_tokens: int = 8000
    rag_top_k: int = 3

    # Metadata
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results from running an experiment."""

    config: ExperimentConfig
    eval_report: EvalReport
    metadata: dict = field(default_factory=dict)


class ExperimentRunner:
    """
    Runs controlled experiments to compare configurations.
    """

    def __init__(self, base_chatbot_func: Callable) -> None:
        """
        Args:
            base_chatbot_func: A callable that creates a chatbot response.
                              Should accept (query, history, config) and
                              return a response string.
        """
        self._base_chatbot_func = base_chatbot_func
        self._results_dir = Path("data/experiments")
        self._results_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(
        self,
        config: ExperimentConfig,
        dataset: EvaluationDataset,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration.
            dataset: Test dataset.

        Returns:
            ExperimentResult with metrics.
        """
        logger.info("Running experiment: %s", config.name)

        # Create a chatbot function with this config
        def configured_chatbot(query: str) -> str:
            return self._base_chatbot_func(query, config)

        # Evaluate
        evaluator = Evaluator(configured_chatbot)
        report = evaluator.evaluate_dataset(dataset)

        # Create result
        result = ExperimentResult(
            config=config,
            eval_report=report,
            metadata={
                "dataset_size": dataset.get_test_cases().__len__(),
            },
        )

        # Save result
        self._save_result(result)

        return result

    def run_ablation_study(
        self,
        dataset: EvaluationDataset,
        base_config: ExperimentConfig = None,
    ) -> list[ExperimentResult]:
        """
        Run an ablation study by systematically disabling components.

        Tests the impact of:
        - Memory (with/without)
        - RAG (with/without)
        - Context strategy (window vs ranked)

        Args:
            dataset: Test dataset.
            base_config: Base configuration to ablate from.

        Returns:
            List of ExperimentResult for each configuration.
        """
        if base_config is None:
            base_config = ExperimentConfig(name="base", description="Full system")

        results = []

        # Baseline: everything enabled
        results.append(self.run_experiment(base_config, dataset))

        # Ablation: no memory
        no_memory_config = ExperimentConfig(
            name="no_memory",
            description="Without memory system",
            use_memory=False,
            use_rag=base_config.use_rag,
            use_tools=base_config.use_tools,
            use_ranked_context=base_config.use_ranked_context,
        )
        results.append(self.run_experiment(no_memory_config, dataset))

        # Ablation: no RAG
        no_rag_config = ExperimentConfig(
            name="no_rag",
            description="Without RAG pipeline",
            use_memory=base_config.use_memory,
            use_rag=False,
            use_tools=base_config.use_tools,
            use_ranked_context=base_config.use_ranked_context,
        )
        results.append(self.run_experiment(no_rag_config, dataset))

        # Ablation: no memory and no RAG
        minimal_config = ExperimentConfig(
            name="minimal",
            description="Without memory and RAG",
            use_memory=False,
            use_rag=False,
            use_tools=base_config.use_tools,
            use_ranked_context=base_config.use_ranked_context,
        )
        results.append(self.run_experiment(minimal_config, dataset))

        # Ablation: ranked context
        ranked_config = ExperimentConfig(
            name="ranked_context",
            description="With ranked context strategy",
            use_memory=base_config.use_memory,
            use_rag=base_config.use_rag,
            use_tools=base_config.use_tools,
            use_ranked_context=True,
        )
        results.append(self.run_experiment(ranked_config, dataset))

        return results

    def run_parameter_sweep(
        self,
        dataset: EvaluationDataset,
        parameter_name: str,
        parameter_values: list,
        base_config: ExperimentConfig = None,
    ) -> list[ExperimentResult]:
        """
        Run a parameter sweep to find optimal values.

        Args:
            dataset: Test dataset.
            parameter_name: Name of the parameter to sweep.
            parameter_values: List of values to test.
            base_config: Base configuration.

        Returns:
            List of ExperimentResult for each parameter value.
        """
        if base_config is None:
            base_config = ExperimentConfig(name="sweep_base")

        results = []

        for value in parameter_values:
            # Create config with this parameter value
            config_dict = {
                "name": f"sweep_{parameter_name}_{value}",
                "description": f"Sweeping {parameter_name}={value}",
                "use_memory": base_config.use_memory,
                "use_rag": base_config.use_rag,
                "use_tools": base_config.use_tools,
                "use_ranked_context": base_config.use_ranked_context,
            }

            # Set the specific parameter
            if hasattr(ExperimentConfig, parameter_name):
                config_dict[parameter_name] = value

            config = ExperimentConfig(**config_dict)
            result = self.run_experiment(config, dataset)
            results.append(result)

        return results

    def compare_results(self, results: list[ExperimentResult]) -> dict:
        """
        Compare multiple experiment results.

        Args:
            results: List of ExperimentResult to compare.

        Returns:
            Dict with comparison metrics.
        """
        comparison = {
            "experiments": [],
            "best_latency": None,
            "best_relevance": None,
            "best_expected": None,
        }

        for result in results:
            exp_summary = {
                "name": result.config.name,
                "avg_latency_ms": result.eval_report.avg_latency_ms,
                "avg_relevance": result.eval_report.avg_relevance_score,
                "pct_expected": result.eval_report.pct_contains_expected,
            }
            comparison["experiments"].append(exp_summary)

        # Find best performers
        if comparison["experiments"]:
            comparison["best_latency"] = min(
                comparison["experiments"],
                key=lambda x: x["avg_latency_ms"],
            )["name"]
            comparison["best_relevance"] = max(
                comparison["experiments"],
                key=lambda x: x["avg_relevance"],
            )["name"]
            comparison["best_expected"] = max(
                comparison["experiments"],
                key=lambda x: x["pct_expected"],
            )["name"]

        return comparison

    def _save_result(self, result: ExperimentResult) -> None:
        """Save experiment result to file."""
        filename = f"{result.config.name}.json"
        filepath = self._results_dir / filename

        data = {
            "config": {
                "name": result.config.name,
                "description": result.config.description,
                "use_memory": result.config.use_memory,
                "use_rag": result.config.use_rag,
                "use_tools": result.config.use_tools,
                "use_ranked_context": result.config.use_ranked_context,
                "max_history_turns": result.config.max_history_turns,
                "max_context_tokens": result.config.max_context_tokens,
                "rag_top_k": result.config.rag_top_k,
            },
            "results": {
                "total_queries": result.eval_report.total_queries,
                "avg_latency_ms": result.eval_report.avg_latency_ms,
                "avg_relevance_score": result.eval_report.avg_relevance_score,
                "pct_contains_expected": result.eval_report.pct_contains_expected,
            },
            "metadata": result.metadata,
        }

        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved experiment result to %s", filepath)

    def load_all_results(self) -> list[dict]:
        """Load all experiment results from the experiments directory."""
        results = []

        for filepath in self._results_dir.glob("*.json"):
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                results.append(data)
            except Exception:
                logger.exception("Failed to load result: %s", filepath)

        return results
