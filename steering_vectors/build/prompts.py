"""Strict JSONL contrast-prompt parsing.

Ported from local ``v2-steering-tools`` (ea65b2f through 6678408). Malformed
rows are refused rather than skipped because each prompt set defines its mean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core import canonjson

PROMPT_ROLES = ("system", "user", "assistant")
PREVIEW_LIMIT = 12


class PromptFileError(ValueError):
    """A prompt file is malformed; messages include file and line."""


def check_messages(messages: Any, where: str) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise PromptFileError(f"{where}: 'messages' must be a non-empty list")
    out: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise PromptFileError(f"{where}: messages[{index}] is not an object")
        extra = set(message) - {"role", "content"}
        if extra:
            raise PromptFileError(
                f"{where}: messages[{index}] has unexpected key(s) {sorted(extra)}"
            )
        role, content = message.get("role"), message.get("content")
        if role not in PROMPT_ROLES:
            raise PromptFileError(
                f"{where}: messages[{index}] role {role!r} not in {list(PROMPT_ROLES)}"
            )
        if not isinstance(content, str) or not content:
            raise PromptFileError(
                f"{where}: messages[{index}] 'content' must be a non-empty string"
            )
        out.append({"role": role, "content": content})
    if out[-1]["role"] == "assistant":
        raise PromptFileError(
            f"{where}: last message has role 'assistant'; capture requires the "
            "new assistant generation position"
        )
    return out


def read_prompt_jsonl(path: str | Path) -> list[Any]:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptFileError(f"{file_path}: no such prompt file") from exc
    except UnicodeDecodeError as exc:
        raise PromptFileError(f"{file_path}: not UTF-8 text ({exc})") from exc
    prompts: list[Any] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        where = f"{file_path}:{lineno}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PromptFileError(f"{where}: not valid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise PromptFileError(
                f"{where}: expected a JSON object, got {type(obj).__name__}"
            )
        if set(obj) == {"text"}:
            if not isinstance(obj["text"], str) or not obj["text"]:
                raise PromptFileError(f"{where}: 'text' must be a non-empty string")
            prompts.append(obj["text"])
        elif set(obj) == {"messages"}:
            prompts.append(check_messages(obj["messages"], where))
        else:
            raise PromptFileError(
                f'{where}: expected exactly one key, either "text" or "messages"; '
                f"got {sorted(obj)}"
            )
    if not prompts:
        raise PromptFileError(f"{file_path}: no prompts")
    return prompts


def prompt_messages(prompt: Any) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(message) for message in prompt]


def format_summary(prompts: Sequence[Any]) -> dict[str, Any]:
    kinds = sorted({"text" if isinstance(p, str) else "messages" for p in prompts})
    patterns: dict[str, int] = {}
    for prompt in prompts:
        pattern = "+".join(m["role"] for m in prompt_messages(prompt))
        patterns[pattern] = patterns.get(pattern, 0) + 1
    return {
        "kind": kinds[0] if len(kinds) == 1 else "mixed",
        "n": len(prompts),
        "n_with_system": sum(
            any(message["role"] == "system" for message in prompt_messages(prompt))
            for prompt in prompts
        ),
        "role_patterns": dict(sorted(patterns.items())),
    }


def copy_prompt_file(source: str | Path, destination: str | Path) -> list[Any]:
    prompts = read_prompt_jsonl(source)
    canonjson.write_text_atomic(
        destination, Path(source).read_text(encoding="utf-8")
    )
    return prompts


def token_length_facts(lengths: Iterable[int]) -> dict[str, Any]:
    values = [int(value) for value in lengths]
    if not values or min(values) <= 0:
        raise PromptFileError("prompt token lengths must be non-empty and positive")
    return {
        "prompt_tokens_min": min(values),
        "prompt_tokens_max": max(values),
        "prompt_tokens_mean": round(sum(values) / len(values), 2),
    }


def preview_block(prompts: Sequence[Any], limit: int = PREVIEW_LIMIT) -> str:
    shown = []
    for prompt in prompts[:limit]:
        if isinstance(prompt, str):
            shown.append(json.dumps(prompt, ensure_ascii=False))
        else:
            shown.append(
                "\n".join(f"[{m['role']}] {m['content']}" for m in prompt)
            )
    body = "```\n" + "\n---\n".join(shown) + "\n```"
    if len(prompts) > limit:
        body += f"\n\n_(first {limit} of {len(prompts)}; see the JSONL for all)_"
    return body
