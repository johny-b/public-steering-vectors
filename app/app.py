"""The steering app: a plain chat with a steering condition on top of it.

The page is a conversation rather than an instrument: a vector to steer with, a
strength to steer at, and moving either archives what you were saying and hands
your opening question back to the send box, so the comparison is between two
transcripts rather than between two numbers.

**Only ``send`` generates.** Changing the condition, clearing, rewinding and
restoring all complete without a completion request: they rearrange what is on
screen and what the next request will carry, and then stop. The person decides
when a question is asked, which at these strengths is the difference between
reading a reply and waiting minutes for one nobody asked for.

**Rewinding is an icon on the question itself**, on every question but the opening
one. Pressing it takes that question, its answer and everything after it off the
conversation, and puts the question back in the send box. All of it goes because
each later turn was produced with this exchange in its prompt, so a transcript
that kept them would be a record of answers to a question that is no longer
there. The icon is Gradio's edit control — the only per-message hook it offers —
and the text left in its editor is what comes back in the box, which makes
rephrasing a question that went nowhere a single gesture.

**It computes nothing.** The condition strings, the condition write and the
refusals all belong to ``app/engine.py``, which :mod:`session` loads by path and
this file only ever drives: ``api.state()``, ``api.set_condition()``, and the
streamed turn in :mod:`session` that is ``api.chat()``'s own sequence with the
reply arriving in pieces. Every figure and every label on this page came back
from one of those calls. The state transitions live in :mod:`session`, with no
Gradio in them, so the behaviour this app adds is testable from a script with no
browser.

**The condition is two controls, at the top of the page: a dropdown and a
strength box.** The dropdown lists exactly the vectors the server reported at
``GET /steering/vectors``, so it cannot offer one a request would be refused for.
The strength box is a *text* box, so that what the person typed reaches the
parser: a comma is a decimal point here, because half the world writes ``0,3``
and an HTML number input would report that field as empty rather than as
mistyped.

**There is nothing to press to apply either of them.** Both are read off the
page at the moment send is pressed, the way the sampling parameters always have
been, so what is on screen is what goes on the wire and there is no state in
between for a forgotten button to leave stale. A strength the box cannot read
refuses the send and says why, rather than falling back on the last good value.
The line under the controls is ``format_condition``'s string for the pair that
will be sent next — or, when the box is unreadable, the reason nothing will be.

**The conversation is archived as the condition moves**, not held over until the
next send: the transcript clears while you are the one changing it, so the page
never shows you turns your next question will not be a continuation of. Typing a
strength out digit by digit archives once rather than once per digit, because
the first keystroke that moves the condition takes the conversation with it and
the rest land on an empty one. The archived conversation's opening question
comes back to the send box, ready to be asked again under the new condition —
unless something is already typed there, which is never worth taking away.

It is not a read-back. The engine holds neither half — both travel in
``vllm_xargs`` on each request — so there is nothing to ask it, and the line is
worded as what this app will send rather than as what the server is serving. The
vector figures in the corner are a different case: they come from the server's
own manifest, and where this checkout holds the same vector the page refused to
start unless the two agreed on the array's digest and on the scale, so those
figures are checked rather than asserted.

**The chain of thought is its own block.** The reply and the thinking stream
separately and are drawn separately — the thinking in a collapsed
``ChatMessage`` with a ``metadata`` title, above the answer it produced — because
they are two different things to read, and because a page that concatenated them
would make a strength that suppresses the answer look like a shorter answer
rather than a reply that is all thinking and no conclusion.

Run it with ``python app/app.py``, next to a server started by ``pod/scripts``.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# ``app/`` is deliberately not a package, so the two modules beside this file are
# reached by putting this directory on the path and importing them by name. The
# repository root goes on too, because ``steering_vectors`` sits in the tree next
# to ``app/`` and ``python app/app.py`` puts only ``app/`` on the path — so
# without this line the page runs from a checkout only when something else has
# already pip-installed the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

import engine as enginelib  # noqa: E402
import session as sessionlib  # noqa: E402

#: A fixed port rather than a free one: an address a person has to discover from
#: a log is an address they cannot put in an ssh tunnel before starting the
#: tunnel. Overridable, because the container this runs in serves its own app tab
#: from a different one (see :func:`main`).
DEFAULT_PORT = 8080

#: Every port this system opens is on the loopback interface by default, and this
#: page in particular authenticates nobody. A literal here rather than a shared
#: constant: nothing else in this repository needs one.
LOOPBACK = "127.0.0.1"

#: Where the host and port can be given without a command line, for the container
#: that starts this page from a supervisor rather than a shell.
HOST_ENV = "APP_HOST"
PORT_ENV = "APP_PORT"

#: One engine, one condition, one session. A per-browser session would be a lie:
#: the strength is this process's, and two tabs each holding "their" strength
#: would each be wrong as soon as the other moved it. The lock serialises the
#: transitions, which is separate from the engine client's own lock over the
#: turn in flight.
SESSION: sessionlib.Session | None = None
LOCK = threading.Lock()

CSS = """
/* The conversation is what the page is for, so it takes the window rather than a
   fixed number of pixels: everything else on the Steering tab moved to the right
   column to leave it the room. The subtraction is the send row and the padding
   Gradio puts around a block. */
.transcript { height: calc(100vh - 300px) !important; min-height: 300px; }
.transcript .bubble-wrap { height: 100% !important; }
/* The opening question carries no rewind icon: rewinding to before it would leave
   a conversation with no beginning, which is what `clear` is for. Gradio offers
   the control on every user message, so the first row's copy of it is hidden
   here. This is presentation only — `Session.delete_from` refuses turn 0 whatever
   the page draws — so if a Gradio release renames these classes the icon comes
   back and says why it cannot be used, rather than quietly starting to work. */
.transcript .message-row:first-child button[aria-label="Edit" i] { display: none; }
/* The right column holds the controls and the archive browser. It is deliberately
   not given its own scrollbar: a scrollbox is where the archive went missing. */
.control-note { font-size: 0.85em; opacity: 0.75; }
/* The list of archived conversations scrolls in place: it grows by one line per
   condition change or clear, and without this it pushes 'restore' off the page. The
   scrollbar is left visible rather than hidden on hover, because a list that can be
   longer than it looks should say so. */
.archive-list { max-height: calc(100vh - 520px); min-height: 140px; overflow-y: auto; }
.archive-list::-webkit-scrollbar { width: 8px; }
.archive-list::-webkit-scrollbar-thumb {
  background: rgba(128, 128, 128, 0.45); border-radius: 4px;
}
/* The archived transcript is read, not glanced at, so it gets real height too. */
.archived { height: calc(100vh - 480px) !important; min-height: 260px; }
.archived .bubble-wrap { height: 100% !important; }
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# rendering what the session holds
# ---------------------------------------------------------------------------


#: What the collapsed thinking block is labelled with in the transcript.
THINKING_TITLE = "chain of thought"

#: How often the page is redrawn while a reply streams. The engine emits a chunk
#: per token — several hundred for one answer — and a redraw per token is a
#: websocket message per token for no gain a person can see. The stream itself is
#: not throttled: every chunk is accumulated, and this only decides how often what
#: has accumulated is drawn.
FRAME_S = 0.08


def rendered(conv, live=None) -> list[tuple[int | None, gr.ChatMessage]]:
    """Every message the transcript draws, tagged with the turn that drew it.

    One walk, read two ways: :func:`transcript` drops the tags and hands the
    messages to the Chatbot, and :func:`turn_at` reads them to answer which turn
    a position in that list belongs to. A turn draws up to three messages — the
    question, the thinking, the answer — so a position in the Chatbot is *not* a
    turn index, and deriving both from one walk is what keeps a click on the
    fourth bubble pointing at the turn that drew it. The streaming turn is tagged
    ``None``: it is not on the conversation yet, so there is nothing to rewind to.
    """
    out: list[tuple[int | None, gr.ChatMessage]] = []
    for index, turn in enumerate(conv.turns):
        out.append((index, gr.ChatMessage(role="user", content=turn.user)))
        if turn.thinking:
            out.append(
                (
                    index,
                    gr.ChatMessage(
                        role="assistant",
                        content=turn.thinking,
                        metadata={"title": THINKING_TITLE, "status": "done"},
                    ),
                )
            )
        elif turn.thinking_error:
            out.append(
                (
                    index,
                    gr.ChatMessage(
                        role="assistant",
                        content=turn.thinking_error,
                        metadata={
                            "title": "no chain of thought — the reader refused this reply"
                        },
                    ),
                )
            )
        out.append((index, gr.ChatMessage(role="assistant", content=turn.assistant)))
    if live is not None:
        out.append((None, gr.ChatMessage(role="user", content=live.user)))
        if live.thinking:
            out.append(
                (
                    None,
                    gr.ChatMessage(
                        role="assistant",
                        content=live.thinking,
                        metadata={"title": THINKING_TITLE, "status": "pending"},
                    ),
                )
            )
        if live.content:
            out.append(
                (None, gr.ChatMessage(role="assistant", content=live.content))
            )
    return out


def transcript(conv, live=None) -> list:
    """The conversation as the Chatbot draws it: thinking above each answer.

    Not the same list as ``conv.messages()``, and deliberately so. That one is the
    history the engine is sent — user text and answers, no thinking, because the
    thinking of a previous turn is not part of the prompt for the next one. This
    one is for reading, and it keeps the two apart on the page instead.

    ``live`` is the turn that is still arriving, appended in the same shape so
    that a streaming reply and a finished one look the same.
    """
    return [message for _, message in rendered(conv, live=live)]


def turn_at(conv, index: int) -> int | None:
    """The turn whose *question* sits at Chatbot position ``index``, if one does.

    ``None`` for anything else — an answer, a thinking block, a position off the
    end, or the turn still arriving — so that a click the page cannot place is
    refused rather than applied to whichever turn happens to be at that number.
    """
    tagged = rendered(conv)
    if not 0 <= index < len(tagged):
        return None
    turn_index, message = tagged[index]
    return turn_index if message.role == "user" else None


def archive_choices(session) -> list[tuple[str, int]]:
    return [(label, index) for index, label in enumerate(session.archive_labels())]


def vector_choices(state) -> list[tuple[str, str]]:
    """The selector's options: every vector the server listed, in its order."""
    return [(vector["label"], vector["id"]) for vector in state["vectors"]]


def condition_readback(session, controls) -> tuple[str, str]:
    """The condition line and the control note, from what the controls are showing.

    Not a read-back of a record any more, and the name is kept only because the
    line is still the one place the page states what will be sent. The controls
    *are* the condition — there is no ``set`` between them and the wire — so this
    renders what is on screen, having first put it through the same
    ``clean_condition`` that a send puts it through. One validator, so the line
    cannot say a strength is fine and the send then refuse it.

    A control the app cannot use is said so in place of a condition, because the
    honest reading of an unusable strength box is "nothing will be sent", and
    that is exactly what happens if send is pressed.
    """
    try:
        cleaned, refusal = session.clean_condition(controls)
        state = session.state()
    except Exception as exc:  # noqa: BLE001
        return f"**unreadable:** {sessionlib.failure(exc)}", gr.update()
    # The header describes whichever vector is selected, and follows the
    # dropdown even while the strength box is unreadable: the two controls fail
    # independently, and freezing the vector's own figures on the previous
    # selection would describe an intervention nothing is about to apply.
    note = header(state, cleaned.get("vector") or controls.get("vector")) + "\n\n"
    if refusal:
        held = state["condition"]
        return (
            f"**not sent:** {refusal}. Nothing goes out while it reads like "
            f"this; the app is still holding {held['vector']} at strength "
            f"{held['strength']:+g}.",
            note + range_note(state),
        )
    return (
        f"**condition:** {enginelib.format_condition(cleaned)}",
        note + range_note(state),
    )


def steering_view(session, status: str, box: str | None = None):
    """Everything the Steering tab shows once a transition has finished.

    Four things, of the page's six. The two it leaves alone are the condition
    controls' own lines, and both absences are the same rule: the condition
    belongs to the controls, and only :func:`on_condition` writes anything that
    describes it.

    The controls themselves are not outputs of anything, because a redraw that
    wrote to them could take back a value a person had typed and had every reason
    to believe was going to be sent. The two lines under them are not outputs of
    *this*, because the controls can move while a reply is arriving: a send that
    re-rendered them when it landed would render them from the condition it went
    out with, and quietly contradict the box. Nothing but an edit changes what
    the next request will carry, so nothing but an edit needs to say so.
    """
    return (
        transcript(session.active),
        gr.update(),
        status,
        gr.update(choices=archive_choices(session), value=None),
        gr.update() if box is None else box,
        gr.update(),
    )


def live_view(session, progress):
    """The Steering tab mid-reply: the transcript grows, nothing else moves."""
    return (
        transcript(session.active, live=progress),
        gr.update(),
        f"streaming — {progress.chunks} chunks so far",
        gr.update(),
        gr.update(),
        gr.update(),
    )


# ---------------------------------------------------------------------------
# the handlers
# ---------------------------------------------------------------------------


def panel(vector, strength, max_tokens, temperature, top_p) -> dict:
    """The whole control panel as one mapping, in the shape the session reads.

    Assembled at the call site of every handler that needs it rather than held
    anywhere: these five values are the request, and reading them off the page
    each time is what makes "what you see is what is sent" true by construction
    rather than by remembering to write a record.
    """
    return {
        "vector": vector,
        "strength": strength,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def views(progresses):
    """A session stream as this tab's outputs, plus the outcome when there is one.

    One yield per redraw. The intermediate frames are rate-limited (see
    :data:`FRAME_S`); the final one is always emitted, because it is the one that
    carries the finished transcript and the status line.
    """
    last = 0.0
    for progress in progresses:
        if progress.done:
            outcome = progress.outcome or sessionlib.Outcome(True, "")
            prefix = "" if outcome.ok else "refused — "
            yield steering_view(SESSION, prefix + outcome.message), outcome
            continue
        now = time.monotonic()
        if now - last < FRAME_S:
            continue
        last = now
        yield live_view(SESSION, progress), None


def on_send(text: str, vector, strength, max_tokens, temperature, top_p):
    """The one handler that generates anything."""
    controls = panel(vector, strength, max_tokens, temperature, top_p)
    with LOCK:
        for view, outcome in views(SESSION.send_stream(text, controls)):
            # The box is emptied while the reply streams and refilled only if the
            # turn was refused, so a refusal leaves what was typed where it was.
            yield view[:4] + (("" if outcome is None or outcome.ok else text),) + view[5:]


def on_regenerate(vector, strength, max_tokens, temperature, top_p):
    """Take the last answer off and draw another one for the same question."""
    controls = panel(vector, strength, max_tokens, temperature, top_p)
    with LOCK:
        for view, outcome in views(SESSION.regenerate_stream(controls)):
            # Nothing to do to the send box: what is being asked again came off the
            # transcript, not out of the box, so whatever is typed there stays.
            yield view[:4] + (gr.update(),) + view[5:]


def on_condition(vector, strength, text):
    """Either control moved: apply it, and archive what it made incomparable.

    Bound to ``input`` on both, so it fires for a value a person changed and not
    for one the page wrote. It applies immediately — that is the whole point of
    there being no ``set`` button — and the conversation is archived here, as the
    condition moves, rather than held over until the next send. Typing a strength
    a digit at a time archives once, not once per digit: the first keystroke that
    moves the condition takes the conversation with it, and the rest land on an
    empty one.

    Two shapes of return, for the same six outputs. Most keystrokes change
    nothing but the two lines under the controls, so everything else is left
    alone — resending the whole transcript on each one would be work for no
    difference. The keystroke that archives redraws the page properly, and hands
    the archived conversation's opening question back to the send box so it can
    be asked again under the new condition. Only if the box is empty: what a
    person has typed there is theirs, and is never worth a convenience.

    Deliberately outside ``LOCK``, for the reason :func:`on_stop` is: a send
    holds that lock for as long as its reply takes to arrive, which at these
    strengths is minutes, and a control that stopped responding to typing for
    the length of a reply would not be a live control. There is nothing here for
    the lock to protect — ``Session.apply`` archives nothing while a reply is
    arriving, and the turn in flight read its condition once, before it started.
    """
    controls = panel(vector, strength, None, None, None)
    outcome = SESSION.apply(controls)
    line, note = condition_readback(SESSION, controls)
    if not outcome.archived:
        return gr.update(), line, gr.update(), gr.update(), gr.update(), note
    return (
        transcript(SESSION.active),
        line,
        outcome.message,
        gr.update(choices=archive_choices(SESSION), value=None),
        gr.update() if (text or "").strip() else outcome.prompt,
        note,
    )


def on_stop() -> str:
    """Interrupt the turn in flight. Deliberately outside ``LOCK``.

    Every other handler takes ``LOCK`` so that two of them cannot move the
    condition at once; this one must run *while* a send is holding it, or it could
    never interrupt anything. It touches no engine state — it raises a flag the
    streaming loop reads between chunks — so there is nothing for the lock to
    protect.
    """
    return SESSION.request_stop().message


def on_edit(data: gr.EditData):
    """The icon on a question: rewind to just before it, with its text back in the box.

    Gradio reports where the message sits in the Chatbot, which is not a turn
    index (see :func:`rendered`), and whatever text was left in the editor. The
    rewind itself is ``Session.delete_from`` — the question, its answer and every
    exchange after it come off, because each of those was produced with this one
    in its prompt — and what lands in the send box is the question again, ready to
    be asked differently. Nothing is sent: as everywhere else on this page, only
    ``send`` generates.

    The text is taken from the editor rather than from the transcript, so a person
    who rewrote the question in place gets their new wording in the box instead of
    the words they were replacing. An emptied editor falls back to the deleted
    question, since a blank box is never what clicking this asked for.
    """
    index = data.index
    if isinstance(index, (tuple, list)):
        # Multimodal content indexes as (message, part); the message is the part
        # of it this page can act on.
        index = index[0] if index else -1
    try:
        position = int(index)
    except (TypeError, ValueError):
        position = -1
    turn_index = turn_at(SESSION.active, position)
    # A refusal leaves the send box alone rather than blanking it: nothing came
    # off the conversation, so there is nothing to hand back, and emptying it
    # would cost whatever was half-typed there.
    if turn_index is None:
        return steering_view(
            SESSION, "refused — that is not a question on this conversation"
        )
    with LOCK:
        outcome = SESSION.delete_from(turn_index)
    if not outcome.ok:
        return steering_view(SESSION, "refused — " + outcome.message)
    typed = data.value if isinstance(data.value, str) else ""
    if typed.strip():
        outcome = sessionlib.Outcome(True, outcome.message, prompt=typed)
    return applied(outcome)


def on_clear():
    """Empty the conversation. The archive keeps it; the screen does not."""
    with LOCK:
        outcome = SESSION.clear()
    return applied(outcome)


def applied(outcome) -> tuple:
    """One redraw after a transition that generated nothing.

    The send box is set from ``outcome.prompt`` — the question a rewind took off
    — so what is in it is always something the person put there or asked for,
    never something that was sent on their behalf. A refusal leaves it alone:
    nothing came off the conversation, so there is nothing to hand back, and
    blanking it would cost whatever was half-typed there.

    Neither condition control, nor either of the lines describing them, is
    touched by anything that comes through here. None of these transitions moves
    the condition.
    """
    prefix = "" if outcome.ok else "refused — "
    return steering_view(
        SESSION,
        prefix + outcome.message,
        box=outcome.prompt if outcome.ok else None,
    )


def on_archive_select(index):
    if index is None:
        return [], ""
    conv = SESSION.archive[int(index)]
    return transcript(conv), f"**recorded at:** {conv.condition()}"


def on_restore(index):
    if index is None:
        return steering_view(SESSION, "select an archived conversation first")
    with LOCK:
        outcome = SESSION.restore(int(index))
        return steering_view(SESSION, outcome.message)


def header(state, vector_id: str | None = None) -> str:
    """The vector and model line, for whichever vector the dropdown is showing.

    Re-rendered on every edit and every finished transition, not written once at
    start-up: the selector moves which vector these figures belong to, and a
    title left over from the previous one would describe an intervention that is
    no longer being applied.

    ``vector_id`` names which of the served vectors to describe, so this can be
    drawn from the dropdown rather than from the app's record — the two agree,
    but the dropdown is the one a person is looking at while they change it.

    The last line says what was checked and when, and it differs per vector
    because the answer does. Every figure here comes from the server's own
    manifest, so it describes the arrays that will actually be added; where this
    checkout holds the same vector, ``choices_from`` refused to start unless the
    two agreed on the array digest and on the scale. Neither is the claim "this
    reply was steered by this vector", which nothing in a completion response
    says.
    """
    vector = next(
        (v for v in state["vectors"] if v["id"] == vector_id), state["vector"]
    )
    server = state["server"]
    if vector["local"]:
        provenance = (
            "Agreed with this checkout at launch, by array digest and by scale."
        )
    else:
        provenance = (
            f"These figures are the server's own: this checkout has no "
            f"vectors/{vector['id']}/ to cross-check them against."
        )
    return (
        f"### {vector['id']} · {vector['name']} · layer {vector['layer']}\n"
        f"{server['model']} at {server['base_url']} · "
        f"‖v‖ {vector['norm']:.4g}, {vector['relative_per_unit'] * 100:.1f}% of the "
        f"stream per raw unit\n\n"
        f"strength is sent per request, already relative: the server scales this "
        f"vector by {vector['scale']:.6g} before adding it at layer "
        f"{vector['layer']}. {provenance}"
    )


def range_note(state) -> str:
    """What the two controls accept, and what changing them costs.

    Worth saying plainly because the answer to the first is "immediately, and
    there is nothing to press", and the answer to the second is "the
    conversation on screen" — which is a thing to know before you touch the box
    rather than after the transcript has gone.
    """
    bounds = state["range"]
    return (
        f"any strength in {bounds['min']:+.4g} … {bounds['max']:+.4g} can be "
        f"typed, with either `.` or `,` for the decimal point, where 1.0 is a "
        f"delta the size of a typical residual-stream row. Both controls apply "
        f"as you change them — there is nothing to press — and a strength the "
        f"box cannot read refuses the send rather than falling back on the last "
        f"good one. Moving either archives the conversation, under the condition "
        f"that produced it, and puts its opening question back in the send box. "
        f"Nothing is generated until you press send."
    )


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def build(session) -> gr.Blocks:
    state = session.state()
    # What the two condition controls are about to be built holding. The lines
    # that describe them are rendered from this rather than from the record, so
    # the page is drawn the same way on its first frame as on every one after,
    # where the controls are all there is.
    opening_controls = {
        "vector": state["condition"].get("vector"),
        "strength": state["condition"].get("strength"),
    }

    with gr.Blocks(title="steering, live", analytics_enabled=False) as demo:
        with gr.Tabs():
            with gr.Tab("Steering"):
                # A third of the width for the conversation in flight, two thirds
                # for the controls and the archive browser: the archive is read as
                # much as the live conversation is, and it was the thing with no
                # room. Both columns run the full height of the window.
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        # ``editable="user"`` is what puts an icon on each
                        # question. It is Gradio's edit affordance, and the only
                        # per-message hook it offers, but what the page does with
                        # it is rewind: the click reports which message it came
                        # from, and `on_edit` takes that message and everything
                        # after it off. Editing a question in place and continuing
                        # as though the old answer had been given to the new
                        # question would be a transcript of something that never
                        # happened.
                        chat = gr.Chatbot(
                            value=transcript(session.active),
                            label="conversation",
                            elem_classes=["transcript"],
                            editable="user",
                        )
                        box = gr.Textbox(
                            placeholder="say something",
                            show_label=False,
                            lines=3,
                            max_lines=12,
                            autofocus=True,
                        )
                        with gr.Row():
                            send = gr.Button(
                                "send", variant="primary", scale=1, size="sm",
                                min_width=60,
                            )
                            stop = gr.Button(
                                "stop", variant="stop", scale=1, size="sm",
                                min_width=60,
                            )
                            regen = gr.Button(
                                "regenerate", scale=1, size="sm", min_width=60
                            )
                            clear = gr.Button(
                                "clear", scale=1, size="sm", min_width=60
                            )
                        status = gr.Markdown("")
                    with gr.Column(scale=2, elem_classes=["controls"]):
                        # The condition on the left of the wide column, what a reply
                        # is drawn with on the right, the archive browser under both.
                        with gr.Row():
                            with gr.Column(scale=1, min_width=260):
                                # The other half of the condition, above the
                                # strength because it is the coarser choice: the
                                # strength is a position on a ladder, and which
                                # ladder is this.
                                vector_pick = gr.Dropdown(
                                    choices=vector_choices(state),
                                    value=state["condition"].get("vector"),
                                    label="vector",
                                    filterable=False,
                                    interactive=True,
                                )
                                # A text box rather than a number box, so that
                                # `parse_strength` is handed the characters that
                                # were typed. An HTML number input refuses a comma
                                # in most browsers and then reports the field as
                                # empty, which would turn `0,3` into "you typed
                                # nothing" — the one reading that cannot be
                                # explained to the person who typed it. The cost is
                                # the spinner arrows, which a live box has less use
                                # for anyway; the range is enforced by the session,
                                # and it refuses rather than clamps.
                                strength_box = gr.Textbox(
                                    value=f"{state['condition'].get('strength'):g}",
                                    label="strength",
                                    max_lines=1,
                                    autofocus=False,
                                )
                                condition = gr.Markdown(
                                    condition_readback(session, opening_controls)[0]
                                )
                            with gr.Column(scale=1, min_width=260):
                                # The sampling parameters: the other half of the
                                # condition a reply was produced under. Read off the
                                # page on every send, so a turn cannot run under
                                # parameters the page is no longer showing.
                                with gr.Row(equal_height=True):
                                    max_tokens = gr.Number(
                                        value=sessionlib.PARAM_DEFAULTS["max_tokens"],
                                        label="max tokens",
                                        step=100,
                                        scale=1,
                                        min_width=80,
                                    )
                                    temperature = gr.Number(
                                        value=sessionlib.PARAM_DEFAULTS["temperature"],
                                        label="temperature",
                                        step=0.1,
                                        scale=1,
                                        min_width=80,
                                    )
                                    top_p = gr.Number(
                                        value=sessionlib.PARAM_DEFAULTS["top_p"],
                                        label="top p",
                                        step=0.05,
                                        scale=1,
                                        min_width=80,
                                    )
                                control_note = gr.Markdown(
                                    condition_readback(session, opening_controls)[1],
                                    elem_classes=["control-note"],
                                )
                        gr.Markdown("#### Archive")
                        with gr.Row():
                            with gr.Column(scale=1, min_width=240):
                                archive = gr.Radio(
                                    choices=archive_choices(session),
                                    label=None,
                                    show_label=False,
                                    value=None,
                                    elem_classes=["archive-list"],
                                )
                                restore = gr.Button(
                                    "restore as the active conversation", size="sm"
                                )
                                archived_condition = gr.Markdown("")
                            with gr.Column(scale=2, min_width=280):
                                archived = gr.Chatbot(
                                    label="archived transcript",
                                    elem_classes=["archived"],
                                )

        # Neither condition control is an output of anything. They are what the
        # condition *is*, so a redraw that wrote to them could take back a value
        # a person had typed and had every reason to believe was going to be
        # sent. `on_send` and `on_regenerate` reach past the send box by index,
        # so the order here is load-bearing.
        steering_outputs = [chat, condition, status, archive, box, control_note]
        # Read by the two handlers that build a request, and by the one that
        # describes the condition. Nothing else on the page depends on them.
        condition_controls = [vector_pick, strength_box]
        sending = [box, *condition_controls, max_tokens, temperature, top_p]
        send.click(on_send, sending, steering_outputs)
        box.submit(on_send, sending, steering_outputs)
        regen.click(
            on_regenerate,
            [*condition_controls, max_tokens, temperature, top_p],
            steering_outputs,
        )
        chat.edit(on_edit, None, steering_outputs)
        clear.click(on_clear, None, steering_outputs)
        # Only the status line: the send generator owns the other outputs and is
        # still writing to them, and it reports the stop itself when it lands.
        stop.click(on_stop, None, [status])
        # `input` rather than `change` on both, so this fires for a value a
        # person changed and not for one the page wrote. The components are
        # passed as inputs rather than read off the event, so what the handler
        # acts on is the value Gradio resolved for the option — the vector id —
        # and not the label it is displayed under. The send box is an input so
        # the handler can tell whether handing back the archived question would
        # be taking something away.
        for control in condition_controls:
            control.input(on_condition, [*condition_controls, box], steering_outputs)
        archive.change(on_archive_select, [archive], [archived, archived_condition])
        restore.click(on_restore, [archive], steering_outputs)
    return demo


def make_session() -> sessionlib.Session:
    """Resolve the target and build the session — refusals happen here, before the port."""
    return sessionlib.Session(enginelib.Api(enginelib.Target.resolve()))


def launch(*, host: str = LOOPBACK, port: int = DEFAULT_PORT) -> int:
    """Resolve the target, build the page, and serve it until interrupted.

    In that order, and it matters: resolving the target is where a server that
    does not answer, a manifest that disagrees with this checkout and an opening
    vector the server does not serve are all refused, and refusing before the
    port is bound means there is never a page on screen showing a plausible
    number nothing could have produced.

    Loopback by default. The page has no authentication; it is reached from
    another machine through an ssh tunnel, which is a decision made by whoever
    opens the tunnel — or by whoever passes ``--host``.
    """
    global SESSION
    SESSION = make_session()
    target = SESSION.api.target
    print(f"serving the steering page on http://{host}:{port}", flush=True)
    print(f"  engine   {target.base_url}", flush=True)
    print(f"  model    {target.model}", flush=True)
    print(f"  vectors  {len(target.vectors)} from the server's manifest", flush=True)
    for vector in target.vectors:
        mark = "*" if vector.id == target.opening else " "
        agreed = "agreed with this checkout" if vector.local else "server only"
        print(
            f"    {mark} {vector.describe()}, scale {vector.scale:.6g} ({agreed})",
            flush=True,
        )
    print("  stop with ctrl-c", flush=True)

    demo = build(SESSION)
    demo.queue()
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,
        ssr_mode=False,
        css=CSS,
        quiet=False,
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get(PORT_ENV) or DEFAULT_PORT)
    )
    parser.add_argument("--host", default=os.environ.get(HOST_ENV) or LOOPBACK)
    args = parser.parse_args(argv)
    return launch(host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
