"""The Streamlit app: `uv run turing-projects-v2 chat`.

MAIN is a ChatGPT-style page: the Conversation so far, the input at the bottom, and answers
streamed with status lines while tools run. The Conversation ID is kept in the URL (`?c=…`), so
reloading the page, also after an app restart, reopens the Conversation.
"""

from uuid import uuid4

import streamlit as st

from turing_projects_v2.agent import chat
from turing_projects_v2.agent.chat import ChatService
from turing_projects_v2.config import load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.ui.answer import AnswerView


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


def main_page() -> None:
    st.title("Job Letter Assistant")
    st.caption("Describe the jobs you are looking for in your own words.")
    state = st.session_state
    if "conversation_id" not in state:
        reopened = st.query_params.get("c")
        state.conversation_id = reopened or uuid4().hex
        # Each message is {"role": "user", "text": ...} or {"role": "assistant", "events": [...]}.
        state.messages = stored_messages(reopened) if reopened else []
        st.query_params["c"] = state.conversation_id

    citations = {}  # the latest Search Results, which [n] markers refer to
    for message in state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["text"])
            else:
                view = AnswerView(citations)
                for event in message["events"]:
                    view.show(event)
                view.finish()
                citations = view.citations

    question = st.chat_input("Ask about jobs, e.g. remote AI engineer roles in Texas")
    if not question:
        return
    state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            service = chat_service()
        except AppError as error:
            st.error(str(error))
            state.messages.pop()
            return
        view = AnswerView(citations)
        events = []
        for event in service.send(state.conversation_id, question):
            events.append(event)
            view.show(event)
        view.finish()
    state.messages.append({"role": "assistant", "events": events})


st.set_page_config(page_title="Job Letter Assistant", page_icon="💼")
try:
    load_settings()
except AppError as error:
    st.error(str(error))
    st.stop()
main_page()
