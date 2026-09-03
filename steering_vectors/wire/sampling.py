"""The sampling contract: what may be sent, and how the answer is read.

**The engine answers HTTP 200 to sampling parameters it does not implement.**
Unknown JSON fields are accepted and silently ignored — no error, no warning,
nothing in the log:

    POST /v1/chat/completions {..., "totally_bogus_param_xyz": 123}  -> 200
    POST /v1/chat/completions {..., "top_kk": 1}                     -> 200
    extra_body={"temperatur": 0}                                     -> 200

So a successful request is not evidence that its parameters took effect. A typo,
or a parameter this build does not implement, produces a perfectly ordinary
response sampled with completely different settings than the caller believes, and
a whole measurement can be run at the wrong temperature and look fine. Two
consequences run through this module:

1. **Parameters are allowlisted, and membership is earned by measurement.** A
   parameter appears in :data:`SUPPORTED` only if it has been demonstrated *on a
   live server* that changing it changes observable behaviour. Each entry
   carries that proof, and the proofs are as
   load-bearing as the names: without them the allowlist looks like an arbitrary
   restriction and gets helpfully widened.
2. **:func:`build_request` is the only sanctioned way to construct a request.**
   Anything that assembles a request dictionary by hand reintroduces the whole
   failure class, because it also has to decide which parameters travel natively
   and which travel inside the extra-body envelope — and putting one on the wrong
   transport is, again, a 200.

The response side is strict for the same reason. Thinking is read from exactly
one field, and a message without it is an error rather than a message without
thinking: see :func:`split_message`.

This module covers the chat route only. There is no scoring route — no second
allowlist, no ``build_completion_request``, no ``tokenize`` and no
``score_span`` — because nothing here scores a fixed continuation: the app
generates.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# --- how a parameter reaches the server ------------------------------------
#: A field the OpenAI-compatible API takes at the top level of the request.
NATIVE = "native"
#: An engine extension. The OpenAI client library only passes these through its
#: ``extra_body`` envelope; sent natively, they are dropped by the client before
#: the request is built, which is invisible from the response.
EXTRA = "extra_body"

# --- what kind of thing it is ----------------------------------------------
SAMPLING = "sampling"  # changes the token distribution or the stopping rule
RESPONSE = "response"  # changes what the response contains, not what was sampled
TEMPLATE = "template"  # changes prompt construction (the chat template)
CACHE = "cache"  # changes what the engine may reuse between requests

#: What every ``proof`` below refers to: one row per parameter of a
#: *chat effect matrix* — a measurement against a running server showing that
#: changing that parameter changes observable behaviour. Nothing in this
#: repository re-runs it, so the proofs are recorded claims and the discipline is
#: on whoever widens the allowlist: measure first, then add the entry.
PROOF_SOURCE = "a chat effect matrix"


@dataclass(frozen=True)
class Param:
    """One allowed request parameter."""

    name: str
    transmit: str
    """:data:`NATIVE` or :data:`EXTRA` — which transport the engine reads it from."""
    kind: str
    """:data:`SAMPLING`, :data:`RESPONSE` or :data:`TEMPLATE`."""
    proof: str
    """The effect-matrix row (see :data:`PROOF_SOURCE`) that proves it takes effect."""
    note: str = ""
    """Measured behaviour a caller needs to know, especially where it surprises."""


def _p(name: str, transmit: str, kind: str, proof: str, note: str = "") -> Param:
    return Param(name=name, transmit=transmit, kind=kind, proof=proof, note=note)


# ---------------------------------------------------------------------------
# The chat-route allowlist.
# ---------------------------------------------------------------------------
SUPPORTED: dict[str, Param] = {
    # -- core sampling, native fields ---------------------------------------
    "temperature": _p(
        "temperature",
        NATIVE,
        SAMPLING,
        "effect matrix row 'temperature'",
        "0.0 is greedy: it takes the engine's argmax path rather than the "
        "top-k/top-p path, and is the only reliable way to ask for determinism "
        "(see the note on top_k).",
    ),
    "top_p": _p(
        "top_p",
        NATIVE,
        SAMPLING,
        "effect matrix row 'top_p'",
        "top_p=0.0001 at temperature 2.0 collapses to a single output: the "
        "top-p mask force-keeps exactly one token, so unlike top_k it breaks "
        "exact logit ties deterministically.",
    ),
    "seed": _p(
        "seed",
        NATIVE,
        SAMPLING,
        "effect matrix row 'seed'",
        "Reproducible sequentially only. Under concurrent load the same seed "
        "gives different output, because the engine's numerics depend on batch "
        "composition; byte-for-byte replay additionally needs a batch width of 1 "
        "and one request in flight. Concurrency is therefore part of a run's "
        "condition, not an implementation detail.",
    ),
    "max_tokens": _p(
        "max_tokens",
        NATIVE,
        SAMPLING,
        "effect matrix row 'max_tokens'",
        "Counts thinking tokens. If it cuts the generation off before thinking "
        "ends, the entire output stays in the thinking field and the content is "
        "empty — which reads as a refusal but is a budget problem. Leave 1-2k "
        "tokens of headroom whenever an answer is expected.",
    ),
    "stop": _p(
        "stop",
        NATIVE,
        SAMPLING,
        "effect matrix row 'stop'",
        "Dangerous with thinking on: stop strings are matched against the "
        "thinking text too, so stop=['\\n\\n'] truncates a few tokens into the "
        "thinking block. Measured: 27 characters of output, finish_reason=stop.",
    ),
    "n": _p(
        "n",
        NATIVE,
        SAMPLING,
        "effect matrix row 'n'",
        "n=4 returns 4 choices. Each choice needs its own thinking read; a "
        "caller that reads choices[0] only is measuring one sample and paying "
        "for four.",
    ),
    "presence_penalty": _p(
        "presence_penalty",
        NATIVE,
        SAMPLING,
        "effect matrix row 'presence_penalty'",
        "Range [-2, 2]; out of range is one of the few genuine HTTP 400s.",
    ),
    "frequency_penalty": _p(
        "frequency_penalty",
        NATIVE,
        SAMPLING,
        "effect matrix row 'frequency_penalty'",
        "Range [-2, 2].",
    ),
    "logit_bias": _p(
        "logit_bias",
        NATIVE,
        SAMPLING,
        "effect matrix row 'logit_bias'",
        "{token_id: bias}, with the ids as JSON strings. Proven by forcing: "
        "{'57': +100} makes the model emit 'ZZZZZ…'. Token ids come from the "
        "server's own tokenizer (see tokenize()), never from a local one.",
    ),
    # -- response shaping, native -------------------------------------------
    "logprobs": _p(
        "logprobs",
        NATIVE,
        RESPONSE,
        "effect matrix row 'logprobs'",
        "A bool on this route (the completions route takes an int — the two are "
        "not interchangeable, and the wrong one is accepted). Needed to see the "
        "top-1/top-2 gap, which is how tie behaviour is observed at all.",
    ),
    "top_logprobs": _p(
        "top_logprobs",
        NATIVE,
        RESPONSE,
        "effect matrix row 'logprobs'",
        "How many alternatives to report; requires logprobs=True.",
    ),
    # -- engine extensions: must travel inside extra_body -------------------
    "top_k": _p(
        "top_k",
        EXTRA,
        SAMPLING,
        "effect matrix row 'top_k'",
        "top_k=1 is NOT greedy on this engine: the top-k mask keeps every token "
        "whose logit equals the maximum, and this model has exact fp32 ties at "
        "one or two positions per generation, which are then sampled uniformly. "
        "An arm configured 'deterministic' with top_k=1 silently varies. Use "
        "temperature=0.",
    ),
    "min_p": _p(
        "min_p",
        EXTRA,
        SAMPLING,
        "effect matrix row 'min_p'",
        "Same tie caveat as top_k: min_p=0.9 at temperature 2.0 narrows 8/8 "
        "distinct outputs to 3/8 and is identical when seeded, but does not "
        "collapse to a single output.",
    ),
    "repetition_penalty": _p(
        "repetition_penalty",
        EXTRA,
        SAMPLING,
        "effect matrix row 'repetition_penalty'",
        "Engine extension, 1.0 = off. A different mechanism from "
        "frequency_penalty and presence_penalty, not a synonym.",
    ),
    "ignore_eos": _p(
        "ignore_eos",
        EXTRA,
        SAMPLING,
        "effect matrix row 'ignore_eos'",
        "Decodes exactly max_tokens regardless of the end-of-sequence token; how "
        "throughput is measured at a fixed output length.",
    ),
    "return_token_ids": _p(
        "return_token_ids",
        EXTRA,
        RESPONSE,
        "effect matrix row 'return_token_ids'",
        "Adds choices[].token_ids. Without it the key is present and null — one "
        "more field whose presence proves nothing.",
    ),
    # -- prompt construction, extra_body ------------------------------------
    "chat_template_kwargs": _p(
        "chat_template_kwargs",
        EXTRA,
        TEMPLATE,
        "effect matrix row 'chat_template_kwargs'",
        "{'enable_thinking': bool}. With False the server returns a null "
        "thinking field, which is a different experimental condition rather than "
        "a formatting choice. build_request() sets it; pass thinking= instead of "
        "writing it out.",
    ),
    # -- prefix-cache namespacing, extra_body -------------------------------
    "cache_salt": _p(
        "cache_salt",
        EXTRA,
        CACHE,
        "vLLM 0.27.1 source rather than a row of the effect matrix: the field is "
        "declared at entrypoints/openai/chat_completion/protocol.py:453, and "
        "v1/core/kv_cache_utils.py:579-580 folds it into the first block's hash, "
        "which hash_block_tokens() then chains through the whole prefix.",
        "Namespaces the prefix cache. A cached block is keyed on its token ids "
        "and this field and nothing else -- vllm_xargs is not in the key -- so "
        "without it two requests that differ only in steering strength are "
        "eligible to share one set of KV blocks, and the second one is served a "
        "prompt the first one steered. steering_salt() builds the value and "
        "with_steering() sets it, so a caller rarely passes it by hand. It is "
        "the one entry here whose effect is not visible in the reply: the proof "
        "above is a source read, and on a server with prefix caching off (the "
        "default for a hybrid model -- vllm/engine/arg_utils.py:2601-2606) it "
        "does nothing at all.",
    ),
}


# ---------------------------------------------------------------------------
# The experiment's sampling values.
# ---------------------------------------------------------------------------
#: The one place the experiment's sampling values are written down. Everything
#: that generates goes out with these unless it deliberately overrides them, and
#: an override is recorded in the run's condition.
#:
#: They are the checkpoint's own recommended thinking-mode settings, and this is
#: the only record of them: :data:`core.modelprofile.PROFILE` deliberately holds
#: identity and shape facts only, so there is no second copy to cross-check
#: against and no second copy to drift.
DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 8192,
}

#: Recommended by the model card, deliberately not sent. ``min_p`` stays in
#: :data:`SUPPORTED` with its own proof so it can be passed explicitly; sending a
#: no-op 0 by default would be one more parameter to keep honest for no
#: behavioural gain.
OMITTED_FROM_DEFAULTS = ("min_p",)

#: Greedy decoding. Spelled out once so that no caller reaches for ``top_k=1``,
#: which looks greedy, is accepted, and is not (see the ``top_k`` note).
GREEDY: dict[str, Any] = {"temperature": 0.0}

# A default that is not in the allowlist would be caught only at request time, in
# whichever code path happened to run first.
assert set(DEFAULTS) <= set(SUPPORTED), set(DEFAULTS) - set(SUPPORTED)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class UnsupportedParamError(ValueError):
    """A request parameter that is not in the verified allowlist."""


def _hint(name: str, allowed: Iterable[str]) -> str:
    close = difflib.get_close_matches(name, list(allowed), n=2, cutoff=0.6)
    return f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""


def _refuse(
    unknown: Sequence[str],
    allowed: Mapping[str, Param],
    *,
    route: str,
    remedy: str,
) -> None:
    raise UnsupportedParamError(
        "; ".join(
            f"unsupported {route} parameter {name!r}.{_hint(name, allowed)}"
            for name in unknown
        )
        + f".\nThis engine answers HTTP 200 and silently ignores unknown JSON "
        f"fields, so sending it would not fail — it would quietly not take "
        f"effect.\nTo allow it: {remedy}\n"
        f"Allowed on {route}: {sorted(allowed)}"
    )


def validate(params: Mapping[str, Any]) -> None:
    """Raise :class:`UnsupportedParamError` for any key not in :data:`SUPPORTED`."""
    unknown = [k for k in params if k not in SUPPORTED]
    if unknown:
        _refuse(
            unknown,
            SUPPORTED,
            route="/v1/chat/completions",
            remedy=(
                f"measure it first: add a row to {PROOF_SOURCE} showing on a live "
                f"server that the parameter changes behaviour, then add the entry "
                f"(with that proof) to wire.sampling.SUPPORTED."
            ),
        )


def split_params(params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split validated parameters into native kwargs and the extra-body envelope.

    ``None`` means "not set" and is dropped, so a caller can pass an optional
    parameter through without deciding whether to include the key.
    """
    validate(params)
    native: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        (native if SUPPORTED[key].transmit == NATIVE else extra)[key] = value
    return native, extra


# ---------------------------------------------------------------------------
# Building requests
# ---------------------------------------------------------------------------
def steering_salt(vector_id: str, strength: float) -> str:
    """The prefix-cache namespace for one steering condition.

    The engine hashes a cached block over its token ids and ``cache_salt``, and
    not over ``vllm_xargs``. A sweep asks one prompt at nine strengths and holds
    the prompt byte-identical across them on purpose, so all nine are eligible
    to share one set of KV blocks: the first strength to run computes the
    prompt, the other eight inherit its activations from block 36 up, and only
    their generated tokens carry the strength they asked for. Deriving the salt
    from the condition puts each one in its own namespace.

    Both halves of the condition are in it, because two vectors at one strength
    are two different interventions.
    """
    return f"steer:{vector_id}:{float(strength)}"


def build_request(
    model: str,
    messages: str | Iterable[Mapping[str, Any]],
    *,
    thinking: bool = True,
    defaults: bool = True,
    **params: Any,
) -> dict[str, Any]:
    """Keyword arguments for one chat completion. The only sanctioned builder.

    ``messages`` may be a plain string, which becomes a single user message.
    ``thinking`` sets the chat-template flag; an explicitly passed
    ``chat_template_kwargs`` is merged on top of it, and is validated like any
    other parameter.

    ``defaults=False`` sends only what the caller passed. It exists for the
    checks that measure one parameter at a time, where an unrequested default
    would confound the measurement; anything that generates for a result keeps
    the defaults.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    merged: dict[str, Any] = {}
    if defaults:
        merged.update(DEFAULTS)
    merged.update(params)

    template_kwargs: dict[str, Any] = {"enable_thinking": bool(thinking)}
    explicit = merged.pop("chat_template_kwargs", None)
    if explicit is not None:
        validate({"chat_template_kwargs": explicit})
        template_kwargs.update(explicit)

    native, extra = split_params(merged)
    extra["chat_template_kwargs"] = template_kwargs
    return {"model": model, "messages": list(messages), **native, "extra_body": extra}


# ---------------------------------------------------------------------------
# Reading the response: the shape of it
# ---------------------------------------------------------------------------
class ScoringError(RuntimeError):
    """A reply could not be read, or was incomplete.

    Raised by the caller that has a response object in hand and finds it is not
    shaped like a reply at all — a choice list that is empty, a message that is
    not a mapping, a field that came back the wrong type.
    """


# ---------------------------------------------------------------------------
# Reading the response: the thinking field
# ---------------------------------------------------------------------------
#: The one field thinking is read from. The name is part of the wire contract:
#: this engine version returns thinking in ``message.reasoning``, and a server
#: that does not is a server this system has not been verified against.
THINKING_FIELD = "reasoning"

#: Field names that carried thinking in earlier engine versions. They are
#: recognised so a failure can say which server it is talking to — never accepted:
#: silently reading an older name turns "the wire changed" into a slightly
#: different number, and a fallback chain means no reader ever learns that the
#: condition it measured was not the condition it asked for.
DIAGNOSTIC_THINKING_FIELDS = ("reasoning_content",)


class MissingThinkingError(RuntimeError):
    """The response carried no thinking text under :data:`THINKING_FIELD`.

    Thinking is always on for these experiments, so this is fatal rather than a
    thing to work around. The usual causes are a server started without the
    reasoning parser, and a request that disabled thinking through the chat
    template.
    """


@dataclass(frozen=True)
class Split:
    """A chat message separated into its answer and its thinking."""

    content: str
    thinking: str
    fields_seen: dict[str, str]
    """Field name → what it held, as a description rather than a boolean. The
    thinking field is *always present* on this engine and merely null when
    thinking is off, so "the key exists" is evidence of nothing; a description
    distinguishes absent, null, empty and real text, and that is what makes a
    failure diagnosable from the record alone."""


def _read(message: Any, name: str) -> tuple[bool, Any]:
    """``(present, value)`` for both mappings and response objects.

    Response objects put fields the client library does not know about into a
    separate extras mapping, so a plain attribute lookup misses exactly the
    engine-specific fields this module cares about.
    """
    if isinstance(message, Mapping):
        return (name in message), message.get(name)
    if hasattr(message, name):
        return True, getattr(message, name)
    extra = getattr(message, "model_extra", None) or {}
    if name in extra:
        return True, extra[name]
    return False, None


def describe_field(present: bool, value: Any) -> str:
    """How a field's content is recorded: a count, not a presence flag."""
    if not present:
        return "absent"
    if value is None:
        return "null"
    if not isinstance(value, str):
        return f"non-string {type(value).__name__}"
    return f"{len(value)} chars" if value.strip() else "empty"


def split_message(message: Any, *, require_thinking: bool = True) -> Split:
    """Separate a chat message into ``.content`` and ``.thinking``.

    Thinking is read from :data:`THINKING_FIELD` and nowhere else. If that field
    carries no text, this raises: there is no fallback to another field name and
    no silent empty string, because both convert a changed wire into a plausible
    result.

    ``require_thinking=False`` is for the two callers that legitimately expect
    none — a request that deliberately disabled thinking, and a check whose
    subject is that very state.
    """
    _, content = _read(message, "content")
    content = content or ""

    seen: dict[str, str] = {}
    present, value = _read(message, THINKING_FIELD)
    seen[THINKING_FIELD] = describe_field(present, value)
    thinking = value if isinstance(value, str) and value.strip() else ""

    # Recorded for the failure message only; never a source of thinking text.
    for name in DIAGNOSTIC_THINKING_FIELDS:
        diagnostic_present, diagnostic_value = _read(message, name)
        if diagnostic_present:
            seen[name] = describe_field(diagnostic_present, diagnostic_value)

    if not thinking and require_thinking:
        diagnostic = [n for n in DIAGNOSTIC_THINKING_FIELDS if n in seen]
        # The message deliberately names no engine version and no reasoning
        # parser: nothing in this repository records which of either is running,
        # and a message that guessed would send a reader to check the wrong
        # thing.
        diagnosis = (
            f" The response carries {', '.join(diagnostic)} instead, which is an "
            f"older server: thinking is read from {THINKING_FIELD!r} and nowhere "
            f"else."
            if diagnostic
            else ""
        )
        raise MissingThinkingError(
            f"no thinking text in {THINKING_FIELD!r}: "
            + ", ".join(f"{k}={v}" for k, v in seen.items())
            + f"; content={describe_field(True, content)}.{diagnosis}"
            + " Thinking must be on: serve with the checkpoint's reasoning "
            "parser and do not disable it through chat_template_kwargs."
        )
    return Split(content=content, thinking=thinking, fields_seen=seen)


def get_thinking(message: Any, *, require: bool = True) -> str:
    """The thinking text alone. See :func:`split_message` for the rules."""
