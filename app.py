"""Hugging Face Space entrypoint.

Spaces with Gradio SDK look for app.py at repository root.
"""

from HF_SPACE_APP import demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
