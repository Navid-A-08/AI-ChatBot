from ai_chatbot.context import ConversationHistory


def test_windowed_messages_with_no_history():
    history = ConversationHistory()
    messages = history.windowed_messages(max_turns=6, current_user_message="Hello")

    assert messages == [{"role": "user", "content": "Hello"}]


def test_windowed_messages_includes_full_history_under_the_limit():
    history = ConversationHistory()
    history.add_turn("Hi", "Hello there")
    history.add_turn("What's 2+2?", "4")

    messages = history.windowed_messages(max_turns=6, current_user_message="Thanks")

    assert messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Thanks"},
    ]


def test_windowed_messages_drops_older_turns_beyond_the_limit():
    history = ConversationHistory()
    history.add_turn("turn 1 user", "turn 1 assistant")
    history.add_turn("turn 2 user", "turn 2 assistant")
    history.add_turn("turn 3 user", "turn 3 assistant")

    # Only the most recent 1 turn should survive the window (turn 3,
    # not turn 2 — window keeps the MOST recent turns).
    messages = history.windowed_messages(max_turns=1, current_user_message="turn 4 user")

    assert messages == [
        {"role": "user", "content": "turn 3 user"},
        {"role": "assistant", "content": "turn 3 assistant"},
        {"role": "user", "content": "turn 4 user"},
    ]
    # Confirm older dropped turns really are gone, not just unused.
    assert "turn 1" not in str(messages)
    assert "turn 2" not in str(messages)


def test_windowed_messages_with_zero_max_turns_sends_only_current_message():
    history = ConversationHistory()
    history.add_turn("previous user msg", "previous assistant reply")

    messages = history.windowed_messages(max_turns=0, current_user_message="new message")

    assert messages == [{"role": "user", "content": "new message"}]
