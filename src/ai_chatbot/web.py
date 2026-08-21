"""
Gradio web interface for the AI chatbot.

Run with: python -m ai_chatbot.web
"""

import logging

import gradio as gr

from ai_chatbot.config import get_settings
from ai_chatbot.context import ConversationHistory
from ai_chatbot.llm.base import LLMConfig, default_provider_registry
from ai_chatbot.logging_setup import setup_logging
from ai_chatbot.prompts import load_prompt


def create_chatbot():
    """Initialize the chatbot components."""
    settings = get_settings()
    setup_logging(settings.log_level)

    base_system_prompt = load_prompt("system_prompt")
    history = ConversationHistory()

    # Create LLM provider
    if settings.llm_provider == "nvidia":
        llm_config = LLMConfig(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            max_tokens=1024,
            extra_params={"base_url": settings.nvidia_base_url},
        )
    elif settings.llm_provider == "openai":
        llm_config = LLMConfig(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            max_tokens=1024,
        )
    else:
        llm_config = LLMConfig(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )

    provider = default_provider_registry.create_provider(settings.llm_provider, llm_config)

    # Ingest documents
    chunks = history.rag.ingest_all_documents()

    return history, provider, base_system_prompt, settings


# Initialize once at startup
chatbot_state = {}


def respond(message: str, chat_history: list) -> tuple:
    """Handle a user message and return the response."""
    if not chatbot_state:
        convo_history, provider, system_prompt, settings = create_chatbot()
        chatbot_state["history"] = convo_history
        chatbot_state["provider"] = provider
        chatbot_state["system_prompt"] = system_prompt
        chatbot_state["settings"] = settings

    convo_history = chatbot_state["history"]
    provider = chatbot_state["provider"]
    base_system_prompt = chatbot_state["system_prompt"]
    settings = chatbot_state["settings"]

    # Build system prompt with memory and RAG
    prompt_parts = [base_system_prompt]

    memory_context = convo_history.get_memory_context()
    if memory_context:
        prompt_parts.append("\n\n## Relevant Memory\n\n" + memory_context)

    retrieved_chunks = convo_history.retrieve_documents(message, top_k=settings.rag_top_k)
    rag_context = convo_history.format_retrieved_context(retrieved_chunks)
    if rag_context:
        prompt_parts.append("\n\n" + rag_context)

    system_prompt = "".join(prompt_parts)

    # Get messages
    messages = convo_history.windowed_messages(settings.max_history_turns, message)

    # Get response
    try:
        response = provider.send_message(messages, system_prompt)
        reply = str(response.text) if response.text else "No response"
    except Exception as e:
        reply = f"Error: {str(e)}"

    # Ensure message is a string
    message = str(message) if not isinstance(message, str) else message

    # Store in history
    convo_history.add_turn(message, reply)

    return reply


def reset_chat():
    """Reset the conversation."""
    chatbot_state.clear()
    return [], ""


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="AI Chatbot") as demo:
        gr.Markdown(
            """
            # AI Chatbot
            A context-engineering chatbot with memory, RAG, and tool support.
            """
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                scale=8,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("New Conversation")

        # Event handlers
        def user_message(message, history):
            """Add user message to chat."""
            if not message.strip():
                return history, ""
            history = history + [{"role": "user", "content": message}]
            return history, ""

        def bot_response(history):
            """Generate bot response."""
            if not history:
                return history
            user_msg = history[-1]["content"]
            reply = respond(user_msg, history)
            history = history + [{"role": "assistant", "content": reply}]
            return history

        msg.submit(
            user_message,
            [msg, chatbot],
            [chatbot, msg],
            queue=False,
        ).then(
            bot_response,
            chatbot,
            chatbot,
        )

        send_btn.click(
            user_message,
            [msg, chatbot],
            [chatbot, msg],
            queue=False,
        ).then(
            bot_response,
            chatbot,
            chatbot,
        )

        clear_btn.click(reset_chat, outputs=[chatbot, msg])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
