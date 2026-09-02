"""ChoiceScript text helpers used by the interpreter in `player.py`.

Adapted from https://github.com/aypan17/machiavelli/blob/main/machiavelli/utils.py.
"""

from __future__ import annotations

from re import Match

from .types import GameSourceLine


def get_conditional_group(
    lines: list[GameSourceLine],
) -> tuple[list[GameSourceLine], list[GameSourceLine]]:
    """Split off a whole `*if` / `*elseif` / `*else` chain from the front of `lines`."""
    assert lines[0][0].startswith("*if")
    wspace = lines[0][1]
    i = 1
    while i < len(lines):
        if lines[i][1] > wspace:
            i += 1
        elif lines[i][1] == wspace:
            if lines[i][0].startswith(("*elseif", "*elsif", "*else")):
                i += 1
            else:
                break
        else:
            break
    return lines[:i], lines[i:]


def get_nested_group(
    lines: list[GameSourceLine],
) -> tuple[list[GameSourceLine], list[GameSourceLine]]:
    """Split off the first line plus every consecutive line indented under it."""
    assert len(lines) > 0
    wspace = lines[0][1]
    i = 1
    while i < len(lines) and (lines[i][1] > wspace):
        i += 1
    return lines[:i], lines[i:]


def make_lowercase(m: Match[str]) -> str:
    """`re.sub` replacement lower-casing group 1, leaving quoted text alone."""
    if m.group(1) is None:
        return m.group(0)
    return m.group(1).lower()


def make_underscore(m: Match[str]) -> str:
    """`re.sub` replacement renaming a ChoiceScript keyword variable to `keyword_`."""
    if m.group(1) is None:
        return m.group(0)
    return m.group(1).lower() + "_"


def split_into_params(text: str) -> list[str]:
    """Split a command's arguments on whitespace, keeping quotes and parens together.

    Wrong if the text contains nested quotes, as upstream notes.
    """
    split = []
    parens = 0
    in_quote = False
    curr = ""
    for c in text:
        if c.isspace():
            if parens == 0 and not in_quote:
                split.append(curr)
                curr = ""
                continue
        elif c == "(":
            parens += 1
        elif c == ")":
            parens -= 1
        elif c == '"':
            in_quote = not in_quote
        curr += c
    split.append(curr)
    split_ = []
    for s in split:
        if s.strip() != "":
            split_.append(s.strip())
    return split_
