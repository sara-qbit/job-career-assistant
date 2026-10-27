"""MAIN: a ChatGPT-style page with the Conversation so far, the input at the bottom, and answers
streamed with status lines while tools run."""

import streamlit as st

from turing_projects_v2.agent.chat import PASTED_AD_MESSAGE
from turing_projects_v2.errors import AppError
from turing_projects_v2.ui.answer import AnswerView
from turing_projects_v2.ui.session import chat_service, search_settings

st.title("Job Career Assistant")
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


def take_job_ad() -> None:
    """Move the pasted ad out of its box (emptying it) to be sent on this rerun."""
    state.pending_job_ad = state.job_ad
    state.job_ad = ""


# A job ad is often longer than a chat message may be, so it has its own box.
with st.expander("Paste a job ad", icon=":material/content_paste:"):
    st.text_area(
        "Job ad",
        key="job_ad",
        height=200,
        placeholder="Paste the full text of the job ad from the website",
        label_visibility="collapsed",
    )
    st.button(
        "Send the job ad",
        key="send_job_ad",
        on_click=take_job_ad,
        icon=":material/send:",
    )

job_ad = state.pop("pending_job_ad", None)
question = st.chat_input("Ask about jobs, e.g. remote AI engineer roles in Texas")
if job_ad is not None:
    question = PASTED_AD_MESSAGE
if question:
    state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            service = chat_service()
        except AppError as error:
            st.error(str(error))
            state.messages.pop()
            st.stop()
        view = AnswerView(citations)
        events = []
        for event in service.send(
            state.conversation_id, question, search_settings(), job_ad=job_ad
        ):
            events.append(event)
            view.show(event)
        view.finish()
    state.messages.append({"role": "assistant", "events": events})
