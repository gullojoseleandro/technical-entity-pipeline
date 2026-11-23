"""Async LCEL pipeline with bounded retries for invalid structured output."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from schemas import TechnicalEntities


LOGGER = logging.getLogger(__name__)

FORMAT_INSTRUCTIONS = (
    "Return only the fields defined by the structured output schema. "
    "List only technologies explicitly named in the source; use an empty list otherwise. "
    "Set criticality to alta for explicit outages or severe production impact, "
    "baja for clearly low-impact descriptions, and media when impact is uncertain. "
    "Summarize only facts supported by the source."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract technical entities from the user's text. {format_instructions}"),
        ("human", "{text}"),
    ]
)


class ExtractionError(RuntimeError):
    """Configuration failure or exhausted structured-output retries."""


class InvalidStructuredOutput(ValueError):
    """A malformed, incomplete, or truncated model response."""


def build_model(provider: str | None = None) -> ChatOpenAI | ChatAnthropic:
    """Select a provider using the same environment convention as module 1."""
    load_dotenv()
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ExtractionError("Set OPENAI_API_KEY in .env or the environment.")
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ExtractionError("Set ANTHROPIC_API_KEY in .env or the environment.")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            temperature=0,
        )
    raise ExtractionError("LLM_PROVIDER must be 'openai' or 'anthropic'.")


def _validate_response(response: dict[str, Any]) -> TechnicalEntities:
    raw = response.get("raw")
    metadata = getattr(raw, "response_metadata", {}) or {}
    finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if finish_reason in {"length", "max_tokens"}:
        LOGGER.warning("Model output was truncated; retry policy will request another response")
        raise InvalidStructuredOutput("Model output was truncated before completion.")

    if response.get("parsing_error") is not None or response.get("parsed") is None:
        LOGGER.warning("Model returned invalid structured output; retry policy may retry")
        raise InvalidStructuredOutput("Model output could not be parsed or was incomplete.")

    try:
        parsed = TechnicalEntities.model_validate(response["parsed"])
    except ValidationError as error:
        LOGGER.warning("Model output failed schema validation; retry policy may retry")
        raise InvalidStructuredOutput("Model output failed schema validation.") from error
    LOGGER.info("Model output passed Pydantic validation")
    return parsed


def build_chain(model: Any | None = None) -> Runnable:
    """Retry the entire LCEL call once when output is malformed or truncated."""
    selected_model = model if model is not None else build_model()
    structured_model = selected_model.with_structured_output(
        TechnicalEntities, include_raw=True
    )
    chain = PROMPT | structured_model | RunnableLambda(_validate_response)
    return chain.with_retry(
        retry_if_exception_type=(InvalidStructuredOutput, OutputParserException, ValidationError),
        stop_after_attempt=2,
        wait_exponential_jitter=False,
    )


async def process_text(text: str) -> TechnicalEntities:
    """Extract and validate technical entities without blocking the event loop."""
    if not text.strip():
        raise ValueError("Text must not be empty.")
    LOGGER.info("Starting technical entity extraction")
    try:
        result = await build_chain().ainvoke(
            {"text": text, "format_instructions": FORMAT_INSTRUCTIONS}
        )
    except (InvalidStructuredOutput, OutputParserException, ValidationError) as error:
        LOGGER.error("Structured output remained invalid after two attempts")
        raise ExtractionError("Could not validate model output after two attempts.") from error
    LOGGER.info("Technical entity extraction completed")
    return result
