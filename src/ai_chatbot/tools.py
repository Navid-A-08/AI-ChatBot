"""
Tool calling and basic agent capabilities.

Why this exists:
    Tool calling transforms the chatbot from a pure text generator
    into an agent that can take actions in the world. This enables:

    1. External knowledge access: Search the web, query databases
    2. Code execution: Run Python code safely
    3. File operations: Read/write files
    4. API integrations: Call external services

    Phase 6 provides the framework for tool calling with basic
    tools. More advanced tools and agent patterns come in later phases.

How it works:
    Tools are defined as Python functions with JSON schemas describing
    their parameters. When the chatbot receives a query, it can decide
    to call one or more tools instead of (or in addition to) generating
    text. The tool results are then incorporated into the response.

    Claude's native tool use API handles the decision-making about
    when to call tools. We just need to define the tools and handle
    the results.

Alternatives considered:
    - Hardcoded responses: No tool calling, just pattern matching.
      Simple but extremely limited.
    - External tool frameworks (LangChain, etc.): Adds complexity
      and dependencies. For a portfolio project, building it from
      scratch demonstrates understanding.
    - MCP (Model Context Protocol): More modern approach, but
      tool use is sufficient for Phase 6.

Trade-offs accepted:
    - Tool execution is synchronous. Async would be more efficient
      but adds complexity.
    - Limited error handling for now. Production systems need
      robust retry and fallback logic.
    - No sandboxing for code execution. This is a portfolio project,
      not a production system.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Definition of a tool that the chatbot can use."""

    name: str
    description: str
    parameters: dict  # JSON Schema
    function: Callable[..., Any]


@dataclass
class ToolCall:
    """A request to call a tool."""

    tool_name: str
    arguments: dict
    id: str = ""


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_name: str
    content: str
    success: bool = True
    error: str = ""


class ToolRegistry:
    """
    Registry of available tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools_for_api(self) -> list[dict]:
        """
        Get tool definitions in the format expected by Claude's API.

        Returns:
            List of tool definitions for the API call.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            ToolResult with the output or error.
        """
        tool = self._tools.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_call.tool_name,
                content="",
                success=False,
                error=f"Tool not found: {tool_call.tool_name}",
            )

        try:
            logger.info("Executing tool: %s with args: %s", tool.name, tool_call.arguments)
            result = tool.function(**tool_call.arguments)
            return ToolResult(
                tool_name=tool_call.tool_name,
                content=str(result),
                success=True,
            )
        except Exception as e:
            logger.exception("Tool execution failed: %s", tool.name)
            return ToolResult(
                tool_name=tool_call.tool_name,
                content="",
                success=False,
                error=str(e),
            )


def create_calculator_tool() -> Tool:
    """Create a calculator tool for basic math operations."""

    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression safely."""
        # Only allow safe math operations
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "pi": math.pi,
            "e": math.e,
        }

        # Basic security: only allow numbers and safe operations
        import ast
        import operator as op

        # operators
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.BitXor: op.xor,
            ast.USub: op.neg,
        }

        def eval_expr(node):
            if isinstance(node, ast.Num):  # <number>
                return node.n
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                return operators[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                # Handle simple function calls like sqrt(2)
                func_name = node.func.id
                if func_name in allowed_names and callable(allowed_names[func_name]):
                    args = [eval_expr(arg) for arg in node.args]
                    return allowed_names[func_name](*args)
                raise ValueError(f"Function not allowed: {func_name}")
            else:
                raise TypeError(node)

        try:
            tree = ast.parse(expression, mode="eval")
            result = eval_expr(tree.body)
            return str(result)
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

    return Tool(
        name="calculator",
        description="Evaluate mathematical expressions. Supports basic arithmetic (+, -, *, /), powers (^), and common math functions (sqrt, sin, cos, tan, abs, round, min, max). Example: '2 + 2', 'sqrt(16)', 'sin(pi/2)'",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        },
        function=calculator,
    )


def create_current_time_tool() -> Tool:
    """Create a tool to get the current date and time."""

    from datetime import datetime

    def get_current_time(timezone: str = "UTC") -> str:
        """Get the current date and time."""
        now = datetime.now()
        return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} (local time)"

    return Tool(
        name="current_time",
        description="Get the current date and time. Use this when the user asks about the current time, date, or needs timestamp information.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone (currently only local time supported)",
                    "default": "UTC",
                }
            },
            "required": [],
        },
        function=get_current_time,
    )


def create_text_analyzer_tool() -> Tool:
    """Create a tool to analyze text statistics."""

    def analyze_text(text: str) -> str:
        """Analyze text and return statistics."""
        words = text.split()
        sentences = text.split(".")
        paragraphs = text.split("\n\n")

        stats = {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len([p for p in paragraphs if p.strip()]),
            "average_word_length": sum(len(w) for w in words) / max(len(words), 1),
        }

        return json.dumps(stats, indent=2)

    return Tool(
        name="text_analyzer",
        description="Analyze text and return statistics like word count, character count, sentence count, and average word length.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze",
                }
            },
            "required": ["text"],
        },
        function=analyze_text,
    )


def create_default_tool_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()

    # Register default tools
    registry.register(create_calculator_tool())
    registry.register(create_current_time_tool())
    registry.register(create_text_analyzer_tool())

    return registry


class ToolAwareAgent:
    """
    Agent that can use tools to answer queries.

    This is a basic agent that decides when to use tools based on
    the user's query and incorporates tool results into responses.
    """

    def __init__(self, tool_registry: ToolRegistry) -> None:
        self._tool_registry = tool_registry

    def should_use_tools(self, query: str) -> bool:
        """
        Simple heuristic to decide if tools should be used.

        In production, this would be handled by the LLM's tool
        use capability, but for basic cases we can use heuristics.
        """
        tool_keywords = {
            "calculate": ["calculate", "math", "compute", "what is", "how much is", "what's"],
            "time": ["time", "date", "when", "current time", "what day"],
            "analyze": ["analyze", "statistics", "count", "words", "text stats"],
        }

        query_lower = query.lower()

        # Check for math operators
        if any(op in query_lower for op in ["+", "-", "*", "/", "^", "**"]):
            return True

        for category, keywords in tool_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return True

        return False

    def get_relevant_tools(self, query: str) -> list[Tool]:
        """Get tools that might be relevant to the query."""
        query_lower = query.lower()
        relevant = []

        # Check for math-related queries
        math_keywords = ["calculate", "math", "compute", "what is", "how much", "square root", "sqrt", "+" , "-", "*", "/"]
        if any(kw in query_lower for kw in math_keywords):
            calculator = self._tool_registry.get_tool("calculator")
            if calculator:
                relevant.append(calculator)

        # Check for time-related queries
        time_keywords = ["time", "date", "when", "current", "what day"]
        if any(kw in query_lower for kw in time_keywords):
            time_tool = self._tool_registry.get_tool("current_time")
            if time_tool:
                relevant.append(time_tool)

        # Check for text analysis queries
        analysis_keywords = ["analyze", "statistics", "count", "words", "text stats"]
        if any(kw in query_lower for kw in analysis_keywords):
            text_tool = self._tool_registry.get_tool("text_analyzer")
            if text_tool:
                relevant.append(text_tool)

        return relevant
