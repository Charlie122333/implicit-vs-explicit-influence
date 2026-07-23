"""Solver that extracts reasoning content from model responses.

This solver handles extracting internal reasoning from model outputs,
supporting both native thinking blocks (Anthropic) and <think> tags
(DeepSeek, Qwen, etc.).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from inspect_ai._util.content import ContentReasoning
from inspect_ai.solver import Generate, Solver, TaskState, solver

# Regex for <think>...</think> blocks
THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)


def split_think_blocks_from_text(text: str) -> Tuple[Optional[str], str]:
    """Extract <think> blocks from text, returning (internal, external).

    This handles the standard case of <think>...</think> tags, as well as
    edge cases like Qwen sometimes emitting only </think> without opening.

    Args:
        text: The model output text

    Returns:
        Tuple of (internal_reasoning, external_output)
    """
    if not text:
        return None, ""

    # Standard case: find all <think>...</think> blocks
    chunks = THINK_RE.findall(text)
    internal = "\n".join([c.strip() for c in chunks]) if chunks else None
    external = THINK_RE.sub("", text).strip()

    # Handle Qwen edge case: </think> without <think>
    if (
        internal is None
        and "</think>" in text.lower()
        and "<think>" not in text.lower()
    ):
        pos = text.lower().find("</think>")
        maybe_internal = text[:pos].strip()
        rest = text[pos + len("</think>") :].strip()
        if maybe_internal:
            internal = maybe_internal
            external = rest

    return internal, external


def extract_reasoning_from_state(state: TaskState) -> Tuple[Optional[str], str]:
    """Extract reasoning from TaskState output.

    This checks for:
    1. Native ContentReasoning blocks (Anthropic extended thinking)
    2. <think>...</think> tags in the text output

    Args:
        state: The TaskState after model generation

    Returns:
        Tuple of (internal_reasoning, external_output)
    """
    output = state.output

    if not output or not output.message:
        return None, ""

    content = output.message.content
    internal: Optional[str] = None
    external_parts: list[str] = []

    # Handle structured content (list of content blocks)
    if isinstance(content, list):
        reasoning_parts: list[str] = []

        for block in content:
            if isinstance(block, ContentReasoning):
                # Native thinking block
                if not block.redacted:
                    reasoning_parts.append(block.reasoning)
                elif block.summary:
                    # For Claude 4+ models, inspect_ai sets redacted=True but
                    # summary contains the full thinking text (from the API's
                    # thinking field). Use it directly as reasoning.
                    reasoning_parts.append(block.summary)
            elif hasattr(block, "text"):
                external_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "text":
                external_parts.append(getattr(block, "text", ""))

        if reasoning_parts:
            internal = "\n".join(reasoning_parts)

        external_text = "\n".join(external_parts)

    else:
        # Plain string content
        external_text = str(content) if content else ""

    # If no native reasoning found, try parsing <think> tags from external
    if internal is None and external_text:
        tag_internal, external_text = split_think_blocks_from_text(external_text)
        internal = tag_internal

    return internal, external_text


@solver
def extract_reasoning() -> Solver:
    """Extract reasoning content from model output and store in metadata.

    This solver should be used AFTER generation. It extracts internal reasoning
    (from native thinking blocks or <think> tags) and stores it in metadata
    for use by scorers.

    Metadata keys set:
        - internal_reasoning: The extracted reasoning text (or None)
        - external_output: The user-facing output text
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        internal, external = extract_reasoning_from_state(state)

        state.metadata["internal_reasoning"] = internal
        state.metadata["external_output"] = external

        return state

    return solve


@solver
def generate_and_extract() -> Solver:
    """Generate model output and extract reasoning in one step.

    Combines generation with reasoning extraction for convenience.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Generate
        state = await generate(state)

        # Extract reasoning
        internal, external = extract_reasoning_from_state(state)
        state.metadata["internal_reasoning"] = internal
        state.metadata["external_output"] = external

        return state

    return solve
