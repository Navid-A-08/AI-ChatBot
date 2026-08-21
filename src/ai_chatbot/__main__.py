"""Entry point for python -m ai_chatbot"""

import sys

if len(sys.argv) > 1 and sys.argv[1] == "web":
    from ai_chatbot.web import build_ui
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
else:
    from ai_chatbot.cli import run_chat_loop
    run_chat_loop()
