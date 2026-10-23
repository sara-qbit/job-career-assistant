"""LLM-as-a-judge: decides whether a generated statement is supported by its source."""

from typing import Literal, Protocol

from pydantic import BaseModel


class Verdict(BaseModel):
    value: Literal["supported", "fabricated"]
    reason: str


class Judge(Protocol):
    def check(self, statement: str, source: str) -> Verdict: ...
