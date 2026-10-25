"""The Streamlit app: `uv run turing-projects-v2 chat`.

Two pages: MAIN (the chat) and Evaluation (everything behind the answers). The Conversation ID
is kept in the URL (`?c=…`), so reloading the page, also after an app restart, reopens the
Conversation.

The left sidebar starts a new chat, reopens past Conversations, and holds the search choices
(embedding model and reranker) used for every search of the next answer.
"""

from pathlib import Path

import streamlit as st

from turing_projects_v2.config import Settings, load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.ui.session import (
    EMBEDDING_MODEL_LABELS,
    chat_service,
    open_conversation,
)

PAGES = Path(__file__).parent / "app_pages"


def search_choices(settings: Settings) -> None:
    """The sidebar's New chat button and search choices (read back by `search_settings`)."""
    with st.sidebar:
        st.button(
            "New chat",
            key="new_chat",
            icon=":material/edit_square:",
            width="stretch",
            on_click=open_conversation,
            args=(None,),
        )
        models = list(EMBEDDING_MODEL_LABELS)
        st.selectbox(
            "Embedding model",
            models,
            # An unknown name in the settings is reported when the Chat service loads.
            index=models.index(settings.embedding_model)
            if settings.embedding_model in models
            else 0,
            format_func=EMBEDDING_MODEL_LABELS.__getitem__,
            key="embedding_model",
        )
        st.toggle(
            "Reranker",
            value=True,
            key="rerank",
            help="Re-scores the best results with a local cross-encoder.",
        )


def conversation_list() -> None:
    """The past Conversations in the sidebar, most recently used first."""
    try:
        conversations = chat_service().conversations()
    except AppError as error:
        st.sidebar.error(str(error))
        return
    with st.sidebar:
        st.caption("Conversations")
        for conversation in conversations:
            current = conversation.id == st.session_state.conversation_id
            st.button(
                conversation.title,
                key=f"conversation-{conversation.id}",
                type="secondary" if current else "tertiary",
                width="stretch",
                on_click=open_conversation,
                args=(conversation.id,),
            )


st.set_page_config(page_title="Job Letter Assistant", page_icon="💼")
try:
    settings = load_settings()
except AppError as error:
    st.error(str(error))
    st.stop()
if "conversation_id" not in st.session_state:
    open_conversation(st.query_params.get("c"))
else:
    st.query_params["c"] = st.session_state.conversation_id  # also after switching pages
search_choices(settings)
page = st.navigation(
    [
        st.Page(PAGES / "main.py", title="MAIN", icon=":material/chat:", default=True),
        st.Page(PAGES / "evaluation.py", title="Evaluation", icon=":material/analytics:"),
    ],
    position="top",
)
page.run()
# Last, so that a Conversation started by this run's question is listed too.
conversation_list()
