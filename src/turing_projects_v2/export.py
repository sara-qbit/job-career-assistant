"""Conversation export for the Evaluation tab: Markdown to read, JSON to keep or process."""

import json

from turing_projects_v2.agent.chat import CitationsEvent, Conversation, TextEvent, ToolCallEvent


def conversation_markdown(conversation: Conversation, title: str) -> str:
    """Each question with its answer, sources, tool calls and cost, then the total cost."""
    lines = [f"# {title}", ""]
    for number, turn in enumerate(conversation.turns, start=1):
        lines += [f"## {number}. {turn.question}", ""]
        answer = "".join(event.text for event in turn.events if isinstance(event, TextEvent))
        lines += [answer.strip() or "_(no answer text)_", ""]
        sources = [
            result
            for event in turn.events
            if isinstance(event, CitationsEvent)
            for result in event.results
        ]
        if sources:
            lines += ["**Sources:**", ""]
            for result in sources:
                name = result.title + (f" at {result.company}" if result.company else "")
                lines.append(f"{result.number}. [{name}]({result.url})" if result.url else name)
            lines.append("")
        calls = [event for event in turn.events if isinstance(event, ToolCallEvent)]
        if calls:
            lines += ["**Tool calls:**", ""]
            lines += [f"- `{call.name}` {json.dumps(call.args)}" for call in calls]
            lines.append("")
        if turn.usage:
            lines += [f"_{turn.total.describe()}_", ""]
    lines.append(f"**Total:** {conversation.total.describe()}")
    return "\n".join(lines) + "\n"


def conversation_json(conversation: Conversation) -> str:
    """The whole Conversation, including every answer event and token usage record."""
    return conversation.model_dump_json(indent=2)
