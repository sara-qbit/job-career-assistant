"""Security checks the Chat service applies to the candidate's messages before the agent runs."""

import re
from typing import Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from turing_projects_v2.llm import ask_structured
from turing_projects_v2.profile import CandidateProfile
from turing_projects_v2.prompts import load_prompt

MAX_MESSAGE_LENGTH = 2000

# How much of a message the security log keeps.
EXCERPT_LENGTH = 200

type SecurityEventType = Literal["rejected_input", "prompt_injection", "off_topic", "tool_call_cap"]

# Phrases typical of prompt-injection attempts. A match is flagged and logged, not blocked: the
# system prompt's rules still apply, and an honest question may contain such words.
INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules|prompt)\b",
        r"\byou are now\b",
        r"\b(system|hidden|initial) prompt\b",
        r"\b(jailbreak|DAN mode|developer mode)\b",
        r"\bact as (an?|the) (?!job|career|recruit)",
    )
]

PERSONAL_DATA = "[personal data]"

# Control characters other than tab, line feed and carriage return.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def input_problem(message: str) -> str | None:
    """Why the message cannot be processed as given, or None if it can."""
    if not message.strip():
        return "Your message is empty. Please type a question about jobs or applications."
    if len(message) > MAX_MESSAGE_LENGTH:
        return (
            f"Your message has {len(message):,} characters; at most {MAX_MESSAGE_LENGTH:,} "
            "characters are allowed. Please shorten it."
        )
    if CONTROL_CHARACTERS.search(message):
        return "Your message contains control characters. Please type or paste plain text."
    return None


class ScopeVerdict(BaseModel):
    on_topic: bool = Field(description="whether the message belongs in a job search assistant chat")
    reason: str = Field(description="one short sentence")


class ScopeGuard(Protocol):
    def check(self, message: str, config: RunnableConfig | None = None) -> ScopeVerdict: ...


class LLMScopeGuard:
    """Decides with one small structured LLM call whether a message is about job search."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def check(self, message: str, config: RunnableConfig | None = None) -> ScopeVerdict:
        return ask_structured(
            self._llm,
            ScopeVerdict,
            [
                SystemMessage(load_prompt("scope_guard")),
                HumanMessage(f"<candidate_message>\n{message}\n</candidate_message>"),
            ],
            config,
        )


def looks_like_injection(message: str) -> bool:
    return any(pattern.search(message) for pattern in INJECTION_PATTERNS)


def redacted(text: str, profile: CandidateProfile) -> str:
    """`text` with the candidate's personal details replaced, for logs."""
    details = [
        profile.name,
        profile.street,
        profile.postal_code,
        profile.city,
        profile.phone,
        profile.email,
        profile.github,
    ]
    for detail in sorted((detail for detail in details if detail.strip()), key=len, reverse=True):
        text = re.sub(re.escape(detail), PERSONAL_DATA, text, flags=re.IGNORECASE)
    return text
