from ai_chatbot.tools import (
    Tool,
    ToolCall,
    ToolResult,
    ToolRegistry,
    ToolAwareAgent,
    create_calculator_tool,
    create_current_time_tool,
    create_text_analyzer_tool,
    create_default_tool_registry,
)


def test_tool_creation():
    """Test creating a Tool."""
    def dummy_func(x: int) -> int:
        return x * 2

    tool = Tool(
        name="doubler",
        description="Doubles a number",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            }
        },
        function=dummy_func,
    )

    assert tool.name == "doubler"
    assert tool.description == "Doubles a number"


def test_tool_registry():
    """Test ToolRegistry operations."""
    registry = ToolRegistry()

    tool = create_calculator_tool()
    registry.register(tool)

    assert registry.get_tool("calculator") is not None
    assert registry.get_tool("nonexistent") is None


def test_tool_registry_get_tools_for_api():
    """Test getting tools in API format."""
    registry = create_default_tool_registry()
    api_tools = registry.get_tools_for_api()

    assert len(api_tools) == 3  # calculator, current_time, text_analyzer

    # Check format
    for tool_def in api_tools:
        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def


def test_tool_execution():
    """Test executing a tool."""
    registry = create_default_tool_registry()

    tool_call = ToolCall(
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )

    result = registry.execute(tool_call)

    assert result.success is True
    assert result.content == "4"


def test_tool_execution_error():
    """Test tool execution with invalid input."""
    registry = create_default_tool_registry()

    tool_call = ToolCall(
        tool_name="calculator",
        arguments={"expression": "invalid expression !!!"},
    )

    result = registry.execute(tool_call)

    # Should handle error gracefully
    assert result.success is True or "Error" in result.content


def test_tool_execution_not_found():
    """Test executing a non-existent tool."""
    registry = ToolRegistry()

    tool_call = ToolCall(
        tool_name="nonexistent",
        arguments={},
    )

    result = registry.execute(tool_call)

    assert result.success is False
    assert "not found" in result.error.lower()


def test_calculator_tool():
    """Test calculator tool with various expressions."""
    tool = create_calculator_tool()

    # Basic arithmetic
    assert tool.function("2 + 2") == "4"
    assert tool.function("10 - 5") == "5"
    assert tool.function("3 * 4") == "12"
    assert tool.function("10 / 2") == "5.0"

    # Power
    assert tool.function("2 ** 3") == "8"


def test_calculator_tool_functions():
    """Test calculator tool with math functions."""
    tool = create_calculator_tool()

    # sqrt
    result = tool.function("sqrt(16)")
    assert float(result) == 4.0

    # abs
    result = tool.function("abs(-5)")
    assert result == "5"


def test_current_time_tool():
    """Test current time tool."""
    tool = create_current_time_tool()
    result = tool.function()

    assert "Current time:" in result
    assert "2026" in result  # Should have current year


def test_text_analyzer_tool():
    """Test text analyzer tool."""
    tool = create_text_analyzer_tool()

    result = tool.function("Hello world. This is a test.")
    # Should return JSON
    import json
    stats = json.loads(result)

    assert "word_count" in stats
    assert "character_count" in stats
    assert stats["word_count"] == 6


def test_tool_aware_agent():
    """Test ToolAwareAgent."""
    registry = create_default_tool_registry()
    agent = ToolAwareAgent(registry)

    # Should suggest tools for math queries
    assert agent.should_use_tools("Calculate 2 + 2") is True
    assert agent.should_use_tools("What's 10 * 5?") is True

    # Should not suggest tools for general queries
    assert agent.should_use_tools("Tell me a story") is False


def test_get_relevant_tools():
    """Test getting relevant tools for a query."""
    registry = create_default_tool_registry()
    agent = ToolAwareAgent(registry)

    tools = agent.get_relevant_tools("Calculate the square root of 16")
    tool_names = [t.name for t in tools]

    assert "calculator" in tool_names


def test_default_registry_has_all_tools():
    """Test that default registry has all expected tools."""
    registry = create_default_tool_registry()

    assert registry.get_tool("calculator") is not None
    assert registry.get_tool("current_time") is not None
    assert registry.get_tool("text_analyzer") is not None
