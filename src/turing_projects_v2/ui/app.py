"""The Streamlit app: `uv run turing-projects-v2 chat`.

MAIN is a ChatGPT-style page: the Conversation so far, the input at the bottom, and answers
streamed with status lines while tools run. The Conversation ID is kept in the URL (`?c=…`), so
reloading the page, also after an app restart, reopens the Conversation.

The left sidebar starts a new chat, reopens past Conversations, and holds the search choices
(embedding model and reranker) used for every search of the next answer.
"""

from uuid import uuid4

import streamlit as st

from turing_projects_v2.agent import chat
from turing_projects_v2.agent.chat import ChatService, SearchSettings
from turing_projects_v2.config import Settings, load_settings
from turing_projects_v2.errors import AppError
from turing_projects_v2.ui.answer import AnswerView

EMBEDDING_MODEL_LABELS = {
    "gemini": "Gemini Embedding 2",
    "qwen3": "Qwen3-Embedding-8B",
    "minilm": "MiniLM (local)",
}


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
    # Each message is {"role": "user", "text": ...} or {"role": "assistant", "events": [...]}.
    state.messages = stored_messages(conversation_id) if conversation_id else []
    st.query_params["c"] = state.conversation_id


def search_choices(settings: Settings) -> SearchSettings:
    """The sidebar's New chat button and search choices."""
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
        embedding_model = st.selectbox(
            "Embedding model",
            models,
            # An unknown name in the settings is reported when the Chat service loads.
            index=models.index(settings.embedding_model)
            if settings.embedding_model in models
            else 0,
            format_func=EMBEDDING_MODEL_LABELS.__getitem__,
            key="embedding_model",
        )
        rerank = st.toggle(
            "Reranker",
            value=True,
            key="rerank",
            help="Re-scores the best results with a local cross-encoder.",
        )
    return SearchSettings(embedding_model=embedding_model, rerank=rerank)


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


def main_page(search: SearchSettings) -> None:
    st.title("Job Letter Assistant")
    st.caption("Describe the jobs you are looking for in your own words.")
    state = st.session_state

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
        for event in service.send(state.conversation_id, question, search):
            events.append(event)
            view.show(event)
        view.finish()
    state.messages.append({"role": "assistant", "events": events})


st.set_page_config(page_title="Job Letter Assistant", page_icon="💼")
try:
    settings = load_settings()
except AppError as error:
    st.error(str(error))
    st.stop()
if "conversation_id" not in st.session_state:
    open_conversation(st.query_params.get("c"))
main_page(search_choices(settings))
# Last, so that a Conversation started by this run's question is listed too.
conversation_list()
