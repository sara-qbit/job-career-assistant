"""What both pages share: the Chat service, the open Conversation and the search choices.

The open Conversation lives in session state: `conversation_id`, and `messages` as shown on
MAIN, each {"role": "user", "text": ...} or {"role": "assistant", "events": [...]}.
"""

from uuid import uuid4

import streamlit as st

from turing_projects_v2.agent import chat
from turing_projects_v2.agent.chat import ChatService, SearchSettings
from turing_projects_v2.config import load_settings
from turing_projects_v2.errors import AppError


@st.cache_resource(show_spinner="Loading the Knowledge Base …")
def chat_service() -> ChatService:
    return chat.open_chat_service(load_settings())


def stored_messages(conversation_id: str) -> list[dict]:
    """The page messages of a stored Conversation."""
    try:
        conversation = chat_service().conversation(conversation_id)
    except AppError as error:
        st.error(str(error))
        return []
    messages = []
    for turn in conversation.turns:
        messages.append({"role": "user", "text": turn.question})
        messages.append({"role": "assistant", "events": turn.events})
    return messages


def open_conversation(conversation_id: str | None) -> None:
    """Show the stored Conversation `conversation_id`, or a new one if None."""
    state = st.session_state
    state.conversation_id = conversation_id or uuid4().hex
    state.messages = stored_messages(conversation_id) if conversation_id else []
    st.query_params["c"] = state.conversation_id


def search_settings() -> SearchSettings:
    """The search choices made in the sidebar (its widgets' keys)."""
    return SearchSettings(
        embedding_model=st.session_state.embedding_model, rerank=st.session_state.rerank
    )
