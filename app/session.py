"""The app's state transitions, with no Gradio anywhere in them.

The point of this module is that everything the app *decides* — what a typed
strength means, when a conversation is archived, what condition tag it keeps —
can be driven from a script with no browser and no socket, in the same spirit as
``app/engine.py`` keeping :class:`Api` apart from anything that draws it.

**It reimplements nothing.** The condition formatting, the condition write and
every refusal live in ``app/engine.py`` and in the layers under it. This module
only ever *drives* that file's :class:`Api`: every number, every vector id and
every condition label that reaches the screen came out of ``api.state()``,
``api.set_condition()`` or the streamed turn below.

**The controls are the condition.** There is no press of anything between what
the page shows and what goes on the wire: :meth:`Session.clean_condition` reads
the vector and the strength off the panel on every send, exactly as
:func:`clean_params` has always read the sampling parameters, and an unusable
value refuses the send rather than falling back on a remembered one. What the
app keeps in :class:`Api` is a record of that same read, refreshed on every edit
so the labels stay live; it is never a second opinion, because the send-time read
overwrites it before the request is built.

**Archiving follows the condition.** :meth:`Session.split_if_moved` enforces one
rule — a transcript is never continued under a condition that did not produce it
— and :meth:`Session.apply` calls it as the condition moves, so the transcript
clears while you are changing it rather than surprising you at the next send. A
live control passes through several values on the way to the one that was wanted,
but only the first of them finds a conversation to archive: it takes the
conversation with it, and the rest of the keystrokes land on an empty one.
``send_stream`` calls it once more before generating, which covers the edits made
while a reply was still arriving.

**The one thing this module does own is the streamed turn**, because
:meth:`Api.chat` answers in one piece and cannot be watched arriving. It is not a
second way of talking to the engine: :func:`Session.stream_turn` is
:meth:`Api.chat`'s own sequence, step for step and under :meth:`Api._exclusive`,
with the single-shot completion call replaced by a streaming one — the request
built by :func:`wire.sampling.build_request`, the condition put into it by
:func:`engine.with_steering`, and the reassembled message read by
:func:`wire.sampling.split_message`. It returns the same payload
:meth:`Api.chat` returns, field for field. There is exactly one generation path in
this app — :meth:`Session.send` is this generator, drained — so the two cannot
drift apart.

A turn is exactly one request, carrying its own vector and its own strength, so
it cannot straddle a change to either and there is no condition to stamp before
and after it.
"""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steering_vectors import vectorfmt
from steering_vectors.wire import sampling

# ``app/`` is deliberately not a package, so the engine client beside this file is
# reached the way every module in here reaches its neighbours: by putting this
# directory on the path and importing it by name. One name for one file, so there
# is exactly one module object for it however the app was started — which matters,
# because the refusals it raises are matched by class.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as enginelib  # noqa: E402

#: Decimal separators a person may type into the strength box.
#:
#: A comma is accepted because the box is read by whoever is sitting in front of
#: it and half the world writes ``0,3``. It is *translated*, not guessed at: a
#: string carrying both separators, or more than one comma, is refused rather
#: than resolved by a rule about thousands, because ``1,000`` is one number in
#: one convention and another number in the other and this app has no way to
#: know which was meant.
SEPARATORS = ".,"

#: The sampling parameters this page lets a person set, with the defaults it
#: starts them at. Every one of them is in :data:`wire.sampling.SUPPORTED` — each
#: with its own row of the effect matrix proving on a live server that changing it
#: changes behaviour — and they are passed through
#: :func:`wire.sampling.build_request`, which is the only sanctioned builder,
#: rather than assembled into a request here. Nothing was added to the allowlist
#: to make this work.
#:
#: These defaults are the page's, not the repo's: ``sampling.DEFAULTS`` sends
#: temperature 1.0, top_p 0.95, top_k 20, max_tokens 8192 (the model card's
#: recommendation, minus what is deliberately left out). ``top_p`` 1.0 here is
#: therefore a departure from the model's recommendation, asked for deliberately,
#: and ``top_k`` 20 is left as the repo has it because nothing on this page
#: touches it.
PARAM_DEFAULTS: dict[str, float] = {
    "max_tokens": 10000,
    "temperature": 1.0,
    "top_p": 1.0,
}


def clean_params(values: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """The typed sampling parameters, checked, or the refusal saying which is wrong.

    Read off the page on every send rather than held as state: what is in the boxes
    is what the request is built with, so a turn cannot run under parameters the
    page is no longer showing.
    """
    out: dict[str, Any] = {}
    for name, default in PARAM_DEFAULTS.items():
        raw = values.get(name, default)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return {}, f"{name} is empty"
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return {}, f"{name}={raw!r} is not a number"
        if not math.isfinite(number):
            return {}, f"{name}={raw!r} is not a number the engine can be given"
        if name == "max_tokens":
            if number < 1 or number != int(number):
                return {}, f"max_tokens={raw!r} must be a whole number of at least 1"
            out[name] = int(number)
        elif name == "temperature":
            if number < 0:
                return {}, "temperature cannot be negative"
            out[name] = number
        else:
            if not 0 < number <= 1:
                return {}, f"top_p={raw!r} must be greater than 0 and at most 1"
            out[name] = number
    return out, ""


def parse_strength(raw: Any, low: float, high: float) -> tuple[float | None, str]:
    """A strength as a person typed it, or the sentence saying what is wrong.

    The box is a text box rather than a number box precisely so that this
    function gets the characters that were typed. An HTML number input refuses a
    comma in most browsers and reports the field as empty, which would turn
    ``0,3`` into "you typed nothing" — a worse answer than "that is not a
    number", and a much worse one than reading it.

    What is accepted: surrounding space, either separator, a leading sign. What
    is refused, and named:

    * **both separators, or two commas.** ``1,000.5`` and ``1,0,3`` are
      ambiguous, and picking a convention here would silently send a number a
      thousand times off.
    * **anything float() will not take.** No expressions, no percent signs, no
      units — the box holds one number and says so.
    * **infinities and NaN.** ``inf`` parses as a float and would propagate
      through every later block, returning empty output rather than an error.
    * **outside the range.** Refused rather than clamped: moving a typed 2 to 1
      would answer a question that was not asked.

    Never rounded and never snapped. 0.25 typed is 0.25 sent.
    """
    if raw is None:
        return None, "the box is empty"
    if isinstance(raw, bool):
        return None, f"{raw!r} is not a strength"
    if isinstance(raw, (int, float)):
        text = repr(float(raw))
    else:
        text = str(raw).strip()
    if not text:
        return None, "the box is empty"

    if "," in text and "." in text:
        return None, (
            f"{text!r} uses both separators, so which one is the decimal point "
            f"is a guess — write it with one"
        )
    if text.count(",") > 1:
        return None, f"{text!r} has more than one comma in it"
    text = text.replace(",", ".")

    try:
        value = float(text)
    except (TypeError, ValueError):
        return None, f"{text!r} is not a number"
    if not math.isfinite(value):
        return None, f"{text!r} is not a strength that can be sent"
    if not low <= value <= high:
        return None, f"{value:+g} is outside the range {low:+g} … {high:+g}"
    return value, ""


def failure(exc: BaseException) -> str:
    """A failure as one line, from the engine client's own error payload."""
    payload = enginelib.error_payload(exc)
    return f"{payload['error']} ({payload['status']}): {payload['detail']}"


# ---------------------------------------------------------------------------
# what a conversation is
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One exchange, with the condition it was sent under.

    ``condition`` is ``format_condition``'s string; ``vector`` and ``strength``
    are the two values that went into the request, all three taken from the chat
    reply's payload. What a turn records is therefore what this app sent, which
    is the only thing anything here knows: nothing reports back what the engine
    added.
    """

    user: str
    assistant: str
    condition: str
    vector: str | None
    strength: float | None
    finish_reason: str | None = None
    thinking_missing: bool = False
    #: The chain of thought, as the strict reader separated it out of the
    #: reassembled message — shown in the transcript as its own block, never
    #: glued onto the answer, and never sent back to the engine as history.
    thinking: str = ""
    #: What the strict reader said when it refused the reply. Empty when the
    #: reply carried thinking; a sentence when it did not, kept on the turn so
    #: the page can say so instead of drawing an empty block.
    thinking_error: str = ""
    #: How many streamed chunks carried text, and which field each one's text
    #: arrived under. Evidence that the reply streamed, and of where the chain
    #: of thought came from.
    chunks: int = 0
    stream_fields: dict[str, int] = field(default_factory=dict)


@dataclass
class Conversation:
    """A transcript and the condition it was produced under.

    The ``opened_*`` fields are read from the app's record when the conversation
    is created, so that a conversation with no turns in it can still say what it
    is about to run at. Once it has turns, its first turn's stamp is what it is
    tagged with — that is the condition the opening question was actually
    answered in, which is the thing the archive exists to preserve.
    """

    opened_condition: str
    opened_vector: str | None
    opened_strength: float | None
    turns: list[Turn] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.turns

    def messages(self) -> list[dict[str, str]]:
        """The history in the shape ``api.chat`` takes. The UI owns this, not the engine."""
        out: list[dict[str, str]] = []
        for turn in self.turns:
            out.append({"role": "user", "content": turn.user})
            out.append({"role": "assistant", "content": turn.assistant})
        return out

    def first_prompt(self) -> str | None:
        """The opening question of this conversation, or None if it has no turns."""
        return self.turns[0].user if self.turns else None

    def condition(self) -> str:
        return self.turns[0].condition if self.turns else self.opened_condition

    def vector(self) -> str | None:
        return self.turns[0].vector if self.turns else self.opened_vector

    def strength(self) -> float | None:
        return self.turns[0].strength if self.turns else self.opened_strength

    def key(self) -> tuple[str | None, float | None] | None:
        """The pair this conversation's turns ran under, or ``None`` if it has none.

        What :meth:`Session.split_if_moved` compares the next turn against. Taken
        off the first turn rather than off what the controls now say, because the
        question is "what produced the transcript that is already here" — an
        empty conversation has no answer to that and takes any condition.
        """
        return (self.turns[0].vector, self.turns[0].strength) if self.turns else None

    def tag(self) -> str:
        """The short condition tag the archive list is labelled with.

        Both halves, because the archive exists to be compared across and a list
        of strengths with no vector beside them would put two different
        interventions under the same heading.

        It comes off the condition payload of the first turn, not from whatever
        the controls are sitting at now — so an archived conversation keeps the
        condition it ran under even after the page has moved on.
        """
        strength = self.strength()
        if isinstance(strength, (int, float)) and not isinstance(strength, bool):
            shown = f"{float(strength):+g}"
        else:
            shown = repr(strength)
        return f"{self.vector()} at strength {shown}"

    def opening(self, width: int = 60) -> str:
        prompt = self.first_prompt()
        if not prompt:
            return "(no turns)"
        flat = " ".join(prompt.split())
        return flat if len(flat) <= width else flat[: width - 1] + "…"


@dataclass
class Outcome:
    """What one transition did, as the line the page prints.

    ``prompt`` is what the send box should be holding afterwards. A rewind puts
    the question it took off back there, and so does a condition change, which
    hands back the opening question of the conversation it archived — ready to
    be sent, and *only* sent when the person presses send. Nothing in this app
    starts a generation on its own.

    ``archived`` says whether a conversation moved to the archive. The page needs
    it to know whether this transition is worth redrawing the transcript for:
    most edits of the condition change nothing but a label, and resending the
    whole conversation on each keystroke would be work for no difference.
    """

    ok: bool
    message: str
    prompt: str = ""
    archived: bool = False


@dataclass
class Progress:
    """One state of a turn that is still arriving — or the finished turn.

    The page redraws from these while the engine is still decoding. ``payload``
    holds what :meth:`Api.chat` would have returned and is set only on the last
    one; ``outcome`` is the line the page prints and is set only on the last one
    too.
    """

    user: str = ""
    content: str = ""
    thinking: str = ""
    chunks: int = 0
    done: bool = False
    payload: dict[str, Any] | None = None
    outcome: Outcome | None = None


def delta_fields(delta: Any) -> list[tuple[str, str]]:
    """Every piece of text in one streamed delta, under the name it arrived with.

    Nothing here assumes a field name. The client library parks fields it does
    not declare in ``model_extra`` — which is exactly where an engine's thinking
    lands — so reading only the attributes the library knows about is precisely
    how a streamed chain of thought turns into a silent empty string. Everything
    that arrived is collected under its own name and
    :func:`~wire.sampling.split_message` decides afterwards which name counted.
    """
    data: dict[str, Any] = {}
    if isinstance(delta, Mapping):
        data.update(delta)
    else:
        dump = getattr(delta, "model_dump", None)
        if callable(dump):
            data.update(dump())
        data.update(getattr(delta, "model_extra", None) or {})
    return [
        (name, value)
        for name, value in data.items()
        if name != "role" and isinstance(value, str) and value
    ]


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------


class Session:
    """One person's conversations against one engine.

    Deliberately not per-browser. The condition is a property of this process —
    the app's own record of what it will send — so two sessions would be two
    conditions, and the page's labels would each be right about a different one.
    """

    def __init__(self, api: Any) -> None:
        self.api = api
        self.archive: list[Conversation] = []
        self.active = self.open_conversation()
        # Raised by the page's stop button, lowered at the start of every turn. The
        # streaming loop reads it between chunks, so a stopped turn ends the way a
        # finished one does — the stream closed, the lock released, the partial
        # answer recorded — rather than being abandoned mid-flight.
        self._interrupt = threading.Event()
        self.streaming = False

    # -- reading the engine -------------------------------------------------

    def state(self) -> dict[str, Any]:
        return self.api.state()

    def condition_label(self) -> str:
        """The condition the next request carries, by ``format_condition``."""
        return str(self.state()["condition"]["label"])

    def open_conversation(self) -> Conversation:
        condition = self.state()["condition"]
        return Conversation(
            opened_condition=str(condition["label"]),
            opened_vector=condition.get("vector"),
            opened_strength=condition.get("strength"),
        )

    # -- talking ------------------------------------------------------------

    # -- the condition, read off the page ------------------------------------

    def clean_condition(self, values: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
        """The condition the controls are showing, checked, or why it is unusable.

        The counterpart of :func:`clean_params`, and read the same way: off the
        page, on every send, rather than out of a record something pressed a
        button to write. That is what makes the controls the condition instead of
        a request to change it — there is no state between what is on screen and
        what goes on the wire, so there is nothing that can be stale.

        The vector is checked against the set the server reported at startup, so
        a selection cannot become a 400; the strength is checked by
        :func:`parse_strength` against the app's own range.
        """
        try:
            vector = self.api.target.require(
                vectorfmt.vector_id(values.get("vector"))
            ).id
        except Exception as exc:  # noqa: BLE001 - the page must be told which
            return {}, f"vector: {failure(exc)}"

        bounds = self.state()["range"]
        strength, refusal = parse_strength(
            values.get("strength"), float(bounds["min"]), float(bounds["max"])
        )
        if refusal:
            return {}, f"strength: {refusal}"
        return {"vector": vector, "strength": strength}, ""

    def apply(self, values: Mapping[str, Any]) -> Outcome:
        """Write what the controls are showing, and archive what it made incomparable.

        This runs on every edit of either control — every keystroke in the
        strength box — and archiving from here sounds like it would leave a trail
        of conversations behind one decision. It does not, and the reason is that
        the first edit that moves the condition takes the conversation away with
        it: typing ``0.35`` over ``0`` moves the condition once, at ``0.3``, and
        by the time the ``5`` arrives the active conversation is empty and
        :meth:`split_if_moved` has nothing left to do. One archived conversation
        per decision, however many keystrokes the decision took.

        What it costs is that changing your mind mid-edit has already archived
        the conversation. That is recoverable in one click from the archive
        list, and it buys the thing a deferred archive cannot: the transcript
        clears *as* you change the condition, so the page never shows you turns
        the next question will not be a continuation of.

        Not while a reply is arriving. :meth:`_record` appends the finished turn
        to ``self.active``, so swapping it here would file a turn that ran at the
        old condition under the new one. The record still moves — the edit is
        still applied to the next request — and the send that follows archives
        through the same call.
        """
        cleaned, refusal = self.clean_condition(values)
        if refusal:
            return Outcome(False, refusal)
        try:
            result = self.api.set_condition(cleaned)
        except Exception as exc:  # noqa: BLE001
            return Outcome(False, failure(exc))
        label = str(result["condition"]["label"])
        if self.streaming:
            return Outcome(True, label)
        seed = self.active.first_prompt() or ""
        note = self.split_if_moved(cleaned)
        if not note:
            return Outcome(True, label)
        return Outcome(True, f"{label} — {note}", prompt=seed, archived=True)

    def split_if_moved(self, condition: Mapping[str, Any]) -> str:
        """Archive the conversation if the next turn would not belong to it.

        Called from two places, for one rule: a transcript is never continued
        under a condition that did not produce it. :meth:`apply` calls it as the
        condition moves, which is where a person sees it happen; ``send_stream``
        calls it again before it generates, which catches the changes ``apply``
        could not act on because a reply was arriving.

        An empty conversation is never archived, and that is what makes calling
        this on every keystroke cheap: the first moving edit empties the active
        conversation, so the rest of the edits in one decision find nothing to
        archive.

        ``condition`` is the pair being moved to, passed in rather than read off
        the record, because an edit landing between the write and this call would
        otherwise archive against a condition no turn ever ran under. Live
        controls make that window real, however narrow.
        """
        current = (condition["vector"], condition["strength"])
        previous = self.active.key()
        if previous is None or previous == current:
            return ""
        prior = self.active
        self.archive.append(prior)
        self.active = self.open_conversation()
        return (
            f"the condition moved, so the {len(prior.turns)} exchange(s) before it "
            f"are archived as #{len(self.archive)} at {prior.tag()}"
        )

    # -- talking -------------------------------------------------------------

    def _stream_completion(
        self,
        messages: list[dict[str, Any]],
        params: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> Any:
        """``Api._generate``'s request, asked for a chunk at a time.

        The same builder, the same strength envelope, the same defaults, the same
        absence of a seed: a person asking one question twice wants to see the
        spread, and a fixed seed would return the identical reply and read as an
        intervention with no variance.

        ``stream=True`` is passed to the client call as a separate argument and is
        deliberately *not* added to :data:`wire.sampling.SUPPORTED`. That
        allowlist exists because this engine answers 200 to sampling parameters it
        does not implement, so a parameter earns its entry with a row of the
        effect matrix that proves on a live server that changing it changes
        behaviour. ``stream`` is not that kind of parameter — it changes the
        transport, not the distribution, and whether it took effect is evident
        from observation: either chunks arrive or they do not, and the count is
        carried on every turn. Routing it through the allowlist would widen a list
        whose whole value is that every entry was measured, in order to record
        something no measurement is needed for.

        ``vllm_xargs.steer_vector`` and ``vllm_xargs.steer_strength`` are not in
        that allowlist either, and for a different reason: they are not sampling
        parameters at all. They change the model rather than the sampler, and
        they are proven by the engine's own patch. So they go in the same way
        they go in ``Api._generate`` — through :func:`engine.with_steering`,
        after the sanctioned builder has produced the request, merged into the
        extra-body envelope the builder filled with ``chat_template_kwargs``
        rather than replacing it.
        """
        from openai import OpenAI

        target = self.api.target
        # The condition comes in as `state`, captured once by the caller, rather
        # than read off the api here: the record is live now — every edit of
        # either control writes it, including while this reply is arriving — so
        # a second read could build a request the turn is not labelled with.
        condition = dict(state or self.api._state())
        # The page's parameters go in as keyword arguments to the sanctioned
        # builder, which is what decides for each one whether it travels as a
        # native field or inside the extra-body envelope. Nothing is assembled by
        # hand here, and nothing was added to the allowlist.
        request = enginelib.with_steering(
            sampling.build_request(
                target.model, messages, thinking=True, **dict(params or {})
            ),
            condition["vector"],
            condition["strength"],
        )
        client = OpenAI(
            base_url=target.base_url,
            api_key="EMPTY",
            timeout=enginelib.CHAT_TIMEOUT_S,
            max_retries=0,
        )
        return client.chat.completions.create(**request, stream=True)

    def stream_turn(
        self, messages: list[dict[str, Any]], params: Mapping[str, Any] | None = None
    ) -> Iterator[Progress]:
        """One turn, arriving in pieces: :meth:`Api.chat`'s sequence, streamed.

        Everything that makes ``Api.chat``'s answer a measurement is kept, in its
        order:

        * the vector and the strength are read **once, under the lock**, into
          ``sent``, and that same read both builds the request and stamps the
          turn. This is what lets the controls stay live while a reply arrives:
          the record can move underneath a turn in flight and the turn is neither
          relabelled nor rerouted, because nothing reads the record again;
        * the condition on the payload is the one that was sent, taken from that
          same read: there is nothing to read it back from afterwards, and a turn
          is one request, so it cannot straddle a change;
        * the reassembled message goes through the repo's **strict reader**, which
          reads thinking from :data:`wire.sampling.THINKING_FIELD` and nowhere
          else. A reply with no thinking is an error here as it is everywhere
          else in this system: the error is caught so a strength that silences the
          model is still shown rather than crashing the page, but it is carried on
          the turn and printed, never swallowed into an empty block.

        The lock is held for as long as the caller is iterating, which is the
        truth: the generation is in flight until the last chunk.
        """
        api = self.api
        parts: dict[str, list[str]] = {}
        chunks = 0
        finish_reason: str | None = None
        stopped = False

        with api._exclusive("a streamed turn"):
            sent = api._state()
            self._interrupt.clear()
            self.streaming = True
            stream = self._stream_completion(messages, params, sent)
            try:
                for chunk in stream:
                    choice = (chunk.choices or [None])[0]
                    if choice is None:
                        continue
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason
                    arrived = delta_fields(getattr(choice, "delta", None))
                    if not arrived:
                        continue
                    for name, text in arrived:
                        parts.setdefault(name, []).append(text)
                    chunks += 1
                    yield Progress(
                        content="".join(parts.get("content", ())),
                        thinking="".join(parts.get(sampling.THINKING_FIELD, ())),
                        chunks=chunks,
                    )
                    if self._interrupt.is_set():
                        # The engine is told by closing the stream, in the finally
                        # below; the reason is recorded so a short answer is never
                        # mistaken for one the model chose to end.
                        stopped = True
                        finish_reason = "interrupted"
                        break
            finally:
                self.streaming = False
                self._interrupt.clear()
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - closing is best effort
                    pass

        # The message the engine would have sent in one piece, rebuilt under the
        # names the pieces arrived under — not under the names this app expected.
        message = {name: "".join(pieces) for name, pieces in parts.items()}
        thinking_error = ""
        try:
            split = sampling.split_message(message)
        except sampling.MissingThinkingError as exc:
            thinking_error = str(exc)
            split = sampling.split_message(message, require_thinking=False)

        payload = {
            "content": split.content,
            "thinking": split.thinking,
            "thinking_missing": not split.thinking,
            "thinking_error": thinking_error,
            "fields_seen": dict(split.fields_seen),
            "finish_reason": finish_reason,
            "stopped": stopped,
            "condition": api.condition(sent),
            # What streaming adds to the payload, and nothing else: how many
            # chunks carried text, and how much text arrived under each name.
            "stream": {
                "chunks": chunks,
                "fields": {name: len(text) for name, text in message.items()},
                "thinking_field": sampling.THINKING_FIELD,
            },
        }
        yield Progress(
            content=split.content,
            thinking=split.thinking,
            chunks=chunks,
            done=True,
            payload=payload,
        )

    def send_stream(
        self, text: str, controls: Mapping[str, Any] | None = None
    ) -> Iterator[Progress]:
        """One streamed turn on the active conversation, recorded when it lands.

        ``controls`` is the whole panel — vector, strength and the sampling
        parameters — as it stands on screen at the moment send was pressed. Every
        one of them is read here rather than remembered, so a turn cannot run
        under a condition the page is no longer showing, and an unusable one
        refuses the send instead of quietly generating from the last good value.
        """
        if not text or not text.strip():
            yield Progress(done=True, outcome=Outcome(False, "nothing to send"))
            return
        values = dict(controls or {})
        condition, refusal = self.clean_condition(values)
        if refusal:
            yield Progress(user=text, done=True, outcome=Outcome(False, refusal))
            return
        cleaned, refusal = clean_params(values)
        if refusal:
            yield Progress(user=text, done=True, outcome=Outcome(False, refusal))
            return
        try:
            self.api.set_condition(condition)
        except Exception as exc:  # noqa: BLE001
            yield Progress(user=text, done=True, outcome=Outcome(False, failure(exc)))
            return
        # Usually a no-op: `apply` archived as the condition moved. What reaches
        # here is the change made while a reply was arriving, which `apply` left
        # alone rather than file a turn under a condition it did not run at.
        # Before the history is read either way, so the turns that came out of
        # the previous condition are archived rather than prepended to this
        # one's prompt.
        split = self.split_if_moved(condition)
        try:
            # Through the engine client's own request reader, so a malformed
            # history is refused where every other malformed request is.
            messages = enginelib._messages({"messages": self.active.messages() + [
                {"role": "user", "content": text}
            ]})
        except Exception as exc:  # noqa: BLE001 - the page must be told
            yield Progress(done=True, outcome=Outcome(False, failure(exc)))
            return

        try:
            for progress in self.stream_turn(messages, cleaned):
                if not progress.done:
                    yield Progress(
                        user=text,
                        content=progress.content,
                        thinking=progress.thinking,
                        chunks=progress.chunks,
                    )
                    continue
                yield self._record(text, progress, split)
        except Exception as exc:  # noqa: BLE001 - the page must be told
            yield Progress(user=text, done=True, outcome=Outcome(False, failure(exc)))

    def request_stop(self) -> Outcome:
        """Ask the turn in flight to stop after the chunk it is on.

        Nothing is killed: the loop is told to leave, and it leaves through its own
        exit, so what had arrived by then is kept and recorded as a turn with the
        finish reason ``interrupted``. Said no rather than pretended if nothing is
        generating, because a stop that reports success on an idle app teaches the
        button lies.
        """
        if not self.streaming:
            return Outcome(False, "nothing is being generated")
        self._interrupt.set()
        return Outcome(True, "stopping — keeping what has arrived so far")

    def _record(self, text: str, progress: Progress, split: str = "") -> Progress:
        """Put a finished turn on the active conversation and say what happened.

        ``split`` is :meth:`split_if_moved`'s note, reported here rather than
        when it happened: the archive is a consequence of this turn, and saying
        so beside the turn is what makes the conversation's disappearance legible
        instead of something that happened while you were typing.
        """
        payload = progress.payload or {}
        condition = payload.get("condition") or {}
        stream = payload.get("stream") or {}
        self.active.turns.append(
            Turn(
                user=text,
                assistant=payload.get("content") or "",
                condition=str(condition.get("label")),
                vector=condition.get("vector"),
                strength=condition.get("strength"),
                finish_reason=payload.get("finish_reason"),
                thinking_missing=bool(payload.get("thinking_missing")),
                thinking=payload.get("thinking") or "",
                thinking_error=str(payload.get("thinking_error") or ""),
                chunks=int(stream.get("chunks") or 0),
                stream_fields=dict(stream.get("fields") or {}),
            )
        )
        note = str(condition.get("label"))
        if split:
            note += f" — {split}"
        if payload.get("stopped"):
            note += " — stopped by you; what had arrived is kept"
        if payload.get("thinking_error") and not payload.get("stopped"):
            note += " — warning: the reply carried no chain of thought"
        return Progress(
            user=text,
            content=progress.content,
            thinking=progress.thinking,
            chunks=progress.chunks,
            done=True,
            payload=payload,
            outcome=Outcome(True, note),
        )

    def regenerate_stream(
        self, controls: Mapping[str, Any] | None = None
    ) -> Iterator[Progress]:
        """Ask the last question again, in place, and keep the new answer instead.

        The previous answer is dropped — asked for, and the point: the interesting
        thing about a steered model is the spread over repeated asks, and there is
        no seed, so the same question under the same condition is a fresh draw.

        The whole panel is checked *before* the old turn is removed, so a typo in
        the temperature box or an unreadable strength cannot cost an answer.

        If the condition has moved since that question was answered, this is the
        one gesture that reads as re-asking it under the new one: the earlier
        turns are archived by :meth:`split_if_moved` and the question opens a
        fresh conversation, which is what a comparison between two conditions
        wants to be.
        """
        if not self.active.turns:
            yield Progress(
                done=True,
                outcome=Outcome(False, "nothing to regenerate — no turn was taken"),
            )
            return
        values = dict(controls or {})
        for _cleaned, refusal in (
            self.clean_condition(values),
            clean_params(values),
        ):
            if refusal:
                yield Progress(done=True, outcome=Outcome(False, refusal))
                return
        # Held rather than reached for through `self.active`, which `send_stream`
        # may replace: a refusal must put the answer back where it came from, not
        # onto whatever conversation the archive left behind.
        source = self.active
        dropped = source.turns.pop()
        text = dropped.user
        for progress in self.send_stream(text, values):
            if progress.done and progress.outcome and not progress.outcome.ok:
                # Refused, so nothing replaced the old answer: put it back rather
                # than leave the conversation a turn shorter than it was.
                source.turns.append(dropped)
            yield progress

    def send(self, text: str, controls: Mapping[str, Any] | None = None) -> Outcome:
        """One turn, waited out rather than watched.

        The same generator, drained. There is no second request path: whatever is
        true of a streamed turn is true of this one, because it *is* one.
        """
        outcome = Outcome(False, "the turn produced nothing")
        for progress in self.send_stream(text, controls):
            if progress.done and progress.outcome is not None:
                outcome = progress.outcome
        return outcome

    # -- clearing -----------------------------------------------------------

    def clear(self) -> Outcome:
        """Empty the conversation on screen, keeping it in the archive.

        The transcript is rebuilt from ``self.active`` on every redraw, so the way
        to clear it is to make the active conversation a new empty one — not to
        blank a widget, which the next redraw would refill.
        """
        prior = self.active
        if prior.empty:
            self.active = self.open_conversation()
            return Outcome(True, "the conversation was already empty")
        self.archive.append(prior)
        self.active = self.open_conversation()
        return Outcome(
            True,
            f"cleared — the conversation is in the archive as #{len(self.archive)} "
            f"at {prior.tag()}",
        )

    def delete_from(self, index: int) -> Outcome:
        """Take one message and everything after it off the active conversation.

        The conversation is left exactly as it stood *before* that message was
        asked, which is what "as if it never happened" has to mean: the answer to
        it goes too, and so does every turn that followed, because each of those
        was produced with the deleted exchange in its prompt.

        The first message is not deletable: a conversation whose opening question
        is gone is a different conversation, and there is already a way to start
        one of those (``clear``, which keeps the old one in the archive). Nothing
        is archived here — deleting is the one move on this page that is meant to
        leave no trace, and archiving a fragment on every wrong turn would fill the
        archive with things nobody asked to keep.

        The deleted question comes back as ``Outcome.prompt``, so the send box
        holds it afterwards: rewinding to before a question is nearly always the
        first half of asking a better one, and retyping what you just removed is
        work the page can save you. As everywhere else, it is put there and not
        sent.
        """
        turns = self.active.turns
        if not turns:
            return Outcome(False, "there is nothing to delete")
        if index <= 0:
            return Outcome(
                False,
                "the opening question cannot be deleted — clear the conversation "
                "instead, which keeps it in the archive",
            )
        if index >= len(turns):
            return Outcome(False, f"there is no message #{index + 1} to delete")
        removed = turns[index:]
        del turns[index:]
        following = len(removed) - 1
        tail = f" and the {following} exchange(s) after it" if following else ""
        return Outcome(
            True,
            f"deleted message #{index + 1}{tail} — the conversation is where it "
            f"stood before it was asked ({len(turns)} exchange(s) left)",
            prompt=removed[0].user,
        )

    # -- the archive --------------------------------------------------------

    def archive_labels(self) -> list[str]:
        return [
            f"#{index + 1} · {conv.tag()} · {conv.opening()}"
            for index, conv in enumerate(self.archive)
        ]

    def restore(self, index: int) -> Outcome:
        """Make an archived conversation the active one again.

        The archived copy stays in the archive and keeps its own strength tag: it
        is a record of what was said under a condition, and the app is very
        likely holding a different one now. What is restored is the transcript to
        carry on from, and the page says which condition the continuation will run
        under.
        """
        if not (0 <= index < len(self.archive)):
            return Outcome(False, f"there is no archived conversation #{index + 1}")
        source = self.archive[index]
        current = self.active
        if not current.empty:
            self.archive.append(current)
        restored = self.open_conversation()
        restored.turns = list(source.turns)
        self.active = restored
        return Outcome(
            True,
            f"restored #{index + 1} (recorded at {source.tag()}); "
            f"the next turn {restored.opened_condition}",
        )
