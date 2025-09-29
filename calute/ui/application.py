import gradio as gr

from calute.calute import Calute
from calute.types.agent_types import Agent
from calute.types.messages import MessagesHistory

from .helpers import clear_session, process_message
from .themes import APP_SUBTITLE, APP_TITLE, CSS


def create_application(calute: Calute, agent: Agent = None):
    """Create and configure the Gradio application interface.

    Args:
        calute: Calute instance for managing conversations and processing.
        agent: Optional agent configuration for specialized behavior.

    Returns:
        Configured Gradio Blocks application ready for deployment.
    """
    with gr.Blocks(
        title=APP_TITLE,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")),
        css=CSS,
    ) as application:
        st_calute_msgs = gr.State(MessagesHistory(messages=[]))

        with gr.Column(elem_id="app-header"):
            gr.Markdown(f"# {APP_TITLE}")
            with gr.Row():
                gr.Markdown(f"{APP_SUBTITLE}")

        chatbot = gr.Chatbot(
            type="messages",
            label="",
            height=600,
            render_markdown=True,
            show_label=False,
            elem_classes=["main-chatbot"],
            avatar_images=(None, None),
            show_copy_all_button=True,
            show_copy_button=True,
            show_share_button=True,
        )
        with gr.Column(elem_id="composer-wrap"):
            with gr.Group(elem_id="composer"):
                msg = gr.Textbox(
                    label="",
                    placeholder="Message Calute",
                    elem_id="message-input",
                    lines=3,
                    show_label=False,
                    container=True,
                )
                clear = gr.Button("🗑️", elem_id="attach-btn", size="sm", variant="secondary")
                send = gr.Button("↑", elem_id="send-btn", variant="primary", size="sm")

        # wiring (same behavior as before)
        def on_send(user_text, chat, calute_msgs):
            if not user_text.strip():
                return chat, calute_msgs
            yield from process_message(user_text, chat, calute_msgs, agent=agent, calute=calute)

        send.click(
            on_send,
            inputs=[msg, chatbot, st_calute_msgs],
            outputs=[chatbot, st_calute_msgs],
            queue=True,
        ).then(lambda: "", outputs=msg)

        msg.submit(
            on_send,
            inputs=[msg, chatbot, st_calute_msgs],
            outputs=[chatbot, st_calute_msgs],
            queue=True,
        ).then(lambda: "", outputs=msg)

        clear.click(clear_session, outputs=[chatbot, st_calute_msgs], queue=False)
    return application
