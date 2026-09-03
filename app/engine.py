"""The app's engine client: the condition and one turn.

Everything else in this repository measures the intervention in batches and
writes a table. The app is the one surface where a person moves the strength and
watches, turn by turn, what the intervention is doing to the model's answers.

This module is everything about that which is not a widget — :class:`Target`,
:class:`Api`, the labels and the refusals — kept apart from anything that draws
them so that every decision the app makes is reachable without a browser, and is
tested by calling these methods rather than by driving a page.

**It is a client and nothing else.** The engine (``pod/src/vllm_steering``)
holds no *per-request* steering state: both halves of the condition travel with
each request, in ``vllm_xargs.steer_vector`` and ``vllm_xargs.steer_strength``.
What the server does hold is the set of vectors it can be asked for, which it
lists at :data:`MANIFEST_PATH`. Three consequences, each stated because each is
something a reader may expect this module to offer:

* **The strength label is not read back — it is what will be sent.**
  :func:`format_condition` renders this app's own record of the strength it will
  put in the next request. Nothing on the wire can confirm what the engine did
  with it, so the label claims nothing about that; what the strength does is
  visible only in the reply.
* **There is no read-out.** The projection onto the direction, the delta against
  the stream's own magnitude and the post−pre identity would each need the
  engine's activations. Nothing here can produce them, so the page does not draw
  them.
* **The vectors are the server's list, not this app's.** :meth:`Target.resolve`
  asks the server what it holds and the page offers exactly that. Where the
  checkout has a matching ``vectors/<id>/`` the two are cross-checked, by array
  digest and by scale, and a disagreement is refused; where it does not, the
  server's own description is what the page draws. Neither is a claim about an
  individual reply — nothing in a completion response says which vector was
  added to it — but it is the difference between a page whose numbers were
  asserted and one whose numbers were agreed to by the process that will use
  them.

**The condition is a pair.** Which vector and how strongly, both travelling with
each request, both moved the same way: a change to either archives the
conversation it would have made incomparable and hands its opening question
back. :data:`VECTOR_ID_ENV` chooses which one the page opens on; after that the
selector does.

Layering: this file may import ``core``, ``wire`` and ``vectorfmt``, and does not
import the pod, the evals or the inspect providers. The one library it uses for
generation, the OpenAI client, is what every other chat caller in this system
uses, so the app is not a second way to send a chat request.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from steering_vectors import vectorfmt
from steering_vectors.core import modelprofile
from steering_vectors.wire import sampling

PROFILE = modelprofile.PROFILE

#: How far the strength range reaches, measured in the only unit that is
#: comparable between vectors: ``a·‖v‖ / (mean activation norm at the layer)``,
#: the relative perturbation of :data:`steering_vectors.vectorfmt.STRENGTH_CONVENTION`.
#: One means the delta is as large as a typical residual-stream row at that
#: layer, so the range runs from cancelling a stream's worth of the direction to
#: adding one.
#:
#: A geometric endpoint on purpose. The obvious alternative — bounding the range
#: at the largest strength the model still answers at — would put a number on the
#: page that this app has not measured and cannot measure; what strengths remain
#: usable is what a vector's ladder is for. The range says how big the
#: intervention is, and says nothing about whether it is a good idea.
#:
#: It is also, here, the range in the unit the wire carries. The server scales
#: every vector by ``mean_activation_norm_at_layer / ‖v‖`` before adding it, so
#: the ``steer_strength`` this app sends *is already* the relative perturbation
#: and needs no conversion in either direction (see :func:`strength_range`).
RELATIVE_SPAN = 1.0

#: Long enough for a full thinking budget at a strength that makes the model
#: rambling, which is exactly when a person is watching. No retries: a retried
#: chat request is a second forward pass under a condition that may have moved,
#: and it would be reported as one turn.
CHAT_TIMEOUT_S = 600.0

#: Where the server is. An OpenAI-compatible base URL, which is what every other
#: chat caller in this system holds; the steering manifest is not part of that
#: API and sits beside it rather than under it, so :func:`server_root` is what
#: turns one address into the other.
BASE_URL_ENV = "STEER_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"

#: Where the server lists the vectors it holds, relative to its root.
MANIFEST_PATH = "/steering/vectors"

#: Which of the served vectors the page opens on. Only the opening choice: the
#: selector moves it afterwards. Unset, the page opens on the first vector the
#: server lists — a better default than a literal id, because the server's list
#: is the one thing here that is certainly not stale.
VECTOR_ID_ENV = "STEER_VECTOR_ID"

#: The served model name. Optional: when it is unset the first id the server
#: lists at ``GET /models`` is used, which is also this app's "is the server
#: there" check.
MODEL_ENV = "STEER_MODEL"

#: The request field the condition travels in, and the two keys inside it.
#: Copied from ``inspect_providers/steered_provider/provider.py``, which is this
#: repository's existing statement of how a request carries steering; the engine
#: reads both off every request (``pod/src/vllm_steering/patch.py``). They go
#: together — the server answers 400 to one without the other — and a request
#: carrying neither is unsteered.
XARGS_FIELD = "vllm_xargs"
CACHE_SALT_FIELD = "cache_salt"
VECTOR_ARG = "steer_vector"
STRENGTH_ARG = "steer_strength"


class Refused(RuntimeError):
    """The app will not do this now, and the same request may work later.

    Its own class rather than a status chosen at the call site, so that the reason
    travels with the refusal to whatever is going to show it. Retryable by
    construction: every place that raises it is a lock that is held.
    """


class NoServerError(RuntimeError):
    """No server answered at the address this app was pointed at.

    Classified as 503: nothing answered on the address, which is not this app's
    fault and may succeed on a retry. It is the ``GET /models`` call below that
    raises it.
    """


# ---------------------------------------------------------------------------
# the condition, rendered
# ---------------------------------------------------------------------------


def strength_range() -> dict[str, float]:
    """The app's own strength range, from :data:`RELATIVE_SPAN`. Plainly ±1.

    No conversion happens here, because the server does it. Each served vector
    is scaled by ``mean_activation_norm_at_layer / ‖v‖`` on the way onto the
    device, so what travels on the wire is already the relative perturbation and
    the range in that unit is ±:data:`RELATIVE_SPAN` for every vector. The scale
    is a property of the vector and is derived from its own metadata at both
    ends, which is what :func:`choices_from` compares.

    Hence no vector parameter, and hence a selector that changes the vector does
    not move the range: the whole point of the server scaling each vector by its
    own norm is that ±1 means the same size of intervention on all of them.
    """
    return {"min": -RELATIVE_SPAN, "max": RELATIVE_SPAN}


def format_condition(state: Mapping[str, Any]) -> str:
    """The one-line label for the condition the next request will carry.

    ``state`` is this app's own record of the pair it will send — which vector,
    and how strongly — and the label is worded to claim no more than that. Both
    travel inside each request and there is nothing to read either back from, so
    a label promising what the engine is serving would be a label rendered from
    this app's own input: on the path where the request failed it would be
    indistinguishable from a page that is working. Hence "will send", not "is
    at".

    The percentage is the strength itself. ``vectorfmt.relative_perturbation``
    converts a *raw* strength to a fraction of the stream by multiplying by
    ``‖v‖/mean ‖h‖``; the value here has already been through that conversion, on
    the server, so applying it again would report 13.7% of the intervention that
    is actually being asked for. It is also why the vector's id can sit next to
    the percentage without qualifying it: the figure means the same thing
    whichever vector is named.
    """
    vector = state.get("vector")
    strength = state.get("strength")
    if not isinstance(vector, str) or not vector:
        return f"the vector to send is {vector!r}, which is not an id"
    if not isinstance(strength, (int, float)) or isinstance(strength, bool):
        return f"the strength to send is {strength!r}, which is not a number"
    return (
        f"will send {vector} at strength {float(strength):+g} "
        f"({100 * float(strength):+.1f}% of the layer's recorded mean ‖h‖) "
        f"with the next request"
    )


# ---------------------------------------------------------------------------
# what the app is pointed at
# ---------------------------------------------------------------------------


def server_root(base_url: str) -> str:
    """The server's root, given its OpenAI-compatible base URL.

    ``http://host:8001/v1`` → ``http://host:8001``. The steering manifest is not
    part of the OpenAI API and the server does not serve it under that prefix,
    alongside vLLM's own ``/health`` and ``/tokenize``; this app holds the ``/v1``
    address because that is what the chat client needs, so the two are derived
    from one setting rather than configured twice.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        return root[: -len("/v1")]
    return root


def _get_json(url: str, timeout: float) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise NoServerError(
            f"no server answered {url}: {exc}. Start one with pod/scripts/serve.sh, "
            f"or point this app elsewhere with {BASE_URL_ENV}."
        ) from exc


def list_models(base_url: str, *, timeout: float = 10.0) -> list[str]:
    """The ids the server lists at ``GET {base_url}/models``.

    The app's "is the server there" check, and a weak one, which is not
    pretended otherwise: it says something answered on this address and named a
    model. What it is *not* is the vector check — that is
    :func:`fetch_manifest`, and the two are separate calls because a server can
    be up and serving a model while holding no vector this page can use.
    """
    url = base_url.rstrip("/") + "/models"
    payload = _get_json(url, timeout)
    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, list):
        raise NoServerError(f"{url} did not answer with a model list: {payload!r}")
    return [
        str(item["id"]) for item in data if isinstance(item, Mapping) and "id" in item
    ]


def fetch_manifest(base_url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """What the server says it can steer with, from ``GET /steering/vectors``.

    A 404 here is its own message rather than a generic failure, because it has
    one specific cause worth naming: vLLM loads the endpoint plugin only when
    ``VLLM_PLUGINS`` names it, so a server started without that variable serves
    the model perfectly and holds no manifest at all.
    """
    url = server_root(base_url) + MANIFEST_PATH
    try:
        payload = _get_json(url, timeout)
    except NoServerError as exc:
        if "404" in str(exc):
            raise NoServerError(
                f"{url} answered 404, so the server at {base_url} is not "
                f"serving a steering manifest. It was most likely started "
                f"without VLLM_PLUGINS=steering, which is what makes vLLM load "
                f"the endpoint plugin; pod/scripts/serve.sh sets it."
            ) from exc
        raise
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("vectors"), list
    ):
        raise NoServerError(f"{url} did not answer with a vector list: {payload!r}")
    return dict(payload)


@dataclass(frozen=True)
class Vector:
    """One vector the server offers, as this page describes it.

    Built from the server's manifest, because that describes the arrays that
    will actually be added — the checkout does not steer anything. ``local`` says
    whether a ``vectors/<id>/`` in this tree agreed with it; where one exists it
    is compared and a disagreement is refused (:func:`choices_from`), and where
    none exists the page draws the server's description and says so.
    """

    id: str
    name: str
    description: str
    layer: int
    norm: float
    relative_per_unit: float
    scale: float
    sha256: str
    local: bool

    @property
    def label(self) -> str:
        """What the selector shows: the id first, because that is the handle."""
        return f"{self.id} · {self.name}"

    def describe(self) -> str:
        """One line naming this vector, for a log or a failure message.

        The same shape as :func:`steering_vectors.vectorfmt.describe` minus the
        prompt counts, which the manifest does not carry — a served vector is
        described by what it does to the stream, and how it was derived is a
        question for the vector directory.
        """
        return (
            f"vector {self.id} '{self.name}' at layer {self.layer} "
            f"(‖v‖ {self.norm:.4f}, "
            f"{100 * self.relative_per_unit:.1f}% of the stream per raw unit)"
        )


def _corroborate(entry: Mapping[str, Any], vector: Vector) -> bool:
    """Compare one served vector with the checkout's copy. Refuse a mismatch.

    Returns whether a local copy was found at all. Two comparisons, answering
    different questions:

    * the **sha256** says the server loaded the same array this checkout holds.
      It is the only thing that distinguishes a directory from another one with
      the same id and different contents.
    * the **scale** says a strength on the wire means what this page's label
      says it means. Both sides derive it from the same two norms, so a
      disagreement is a disagreement about the vector, and every percentage the
      page prints would be wrong by the ratio.

    A missing local directory is not a mismatch. The server is serving something
    this checkout has never seen, which is a perfectly ordinary state for a
    checkout that is older than the pod's vectors directory, and refusing to
    show it would hide a vector that works.
    """
    try:
        meta = vectorfmt.read_meta(vectorfmt.dir_for(vector.id))
    except (vectorfmt.VectorFormatError, OSError):
        return False

    if str(entry.get("sha256")) != meta["vector_npy_sha256"]:
        raise vectorfmt.VectorFormatError(
            f"the server serves a vector {vector.id} whose array is not the one "
            f"in this checkout:\n"
            f"  server {entry.get('sha256')}\n"
            f"  local  {meta['vector_npy_sha256']}\n"
            f"Two directories carry the same id and different contents, so every "
            f"figure this page would draw for it describes the wrong one."
        )
    local_scale = vectorfmt.steer_scale(meta)
    if abs(vector.scale - local_scale) > 1e-6 * max(1.0, local_scale):
        raise vectorfmt.VectorFormatError(
            f"the server scales vector {vector.id} by {vector.scale!r}, this "
            f"checkout's metadata by {local_scale:.6f}. The strength on the wire "
            f"is multiplied by the server's number, so the page's percentages "
            f"would be wrong by the ratio of the two."
        )
    return True


def choices_from(manifest: Mapping[str, Any]) -> tuple[Vector, ...]:
    """Every vector the server offers, in the order it listed them.

    The order is the server's, which is id order, and it is also the row order
    the engine gathers by — so the selector reads the same way the log does.
    """
    out: list[Vector] = []
    for entry in manifest["vectors"]:
        if not isinstance(entry, Mapping):
            raise NoServerError(f"the manifest holds a non-object entry: {entry!r}")
        try:
            vector = Vector(
                id=str(entry["id"]),
                name=str(entry["name"]),
                description=str(entry.get("description") or ""),
                layer=int(entry["layer"]),
                norm=float(entry["vector_norm"]),
                relative_per_unit=float(entry["relative_per_unit"]),
                scale=float(entry["scale"]),
                sha256=str(entry["sha256"]),
                local=False,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NoServerError(
                f"the manifest describes a vector this page cannot read: "
                f"{entry!r} ({exc})"
            ) from exc
        out.append(replace(vector, local=_corroborate(entry, vector)))
    if not out:
        raise NoServerError(
            "the server serves no steering vectors, so there is no condition "
            "for this page to move. Restart it with a STEER_VECTOR_DIR that "
            "holds some."
        )
    return tuple(out)


class Target:
    """The one engine, and every vector it will steer with.

    Resolved once, at startup, and agreed with the server while resolving. What
    this class holds is not what the app was told at launch but what the server
    reported, cross-checked against the checkout wherever the checkout has an
    opinion.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        vectors: Sequence[Vector],
        opening: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.vectors = tuple(vectors)
        self.opening = self._resolve_opening(opening)

    def _resolve_opening(self, requested: str | None) -> str:
        """Which vector the page starts on, refusing one the server lacks.

        Refused rather than fallen back from: a page that quietly opened on a
        different vector than it was asked for would be a page whose condition
        nobody chose, and the first thing anybody does with it is read the
        header and believe it.
        """
        if requested is None:
            return self.vectors[0].id
        canonical = vectorfmt.vector_id(requested)
        if self.by_id(canonical) is None:
            raise NoServerError(
                f"{VECTOR_ID_ENV}={requested!r} names vector {canonical}, which "
                f"the server at {self.base_url} does not serve. It serves: "
                f"{', '.join(v.id for v in self.vectors)}."
            )
        return canonical

    def by_id(self, vector_id: str) -> Vector | None:
        return next((v for v in self.vectors if v.id == vector_id), None)

    def require(self, vector_id: str) -> Vector:
        """The served vector with this id, or a refusal naming the alternatives."""
        found = self.by_id(vector_id)
        if found is None:
            raise BadRequest(
                f"{vector_id!r} is not a vector this server serves. It serves: "
                f"{', '.join(v.id for v in self.vectors)}."
            )
        return found

    @classmethod
    def resolve(cls) -> Target:
        base_url = os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        model = os.environ.get(MODEL_ENV)
        if not model:
            served = list_models(base_url)
            if not served:
                raise NoServerError(
                    f"the server at {base_url} lists no models, so there is nothing "
                    f"to address a request to. Set {MODEL_ENV} if it serves one "
                    f"under a name it does not list."
                )
            model = served[0]
        else:
            # Asked anyway: an explicit name skips the lookup, not the check that
            # something is listening, and a page that came up against no server
            # would refuse on the first send instead of at startup.
            list_models(base_url)
        return cls(
            base_url=base_url,
            model=str(model),
            vectors=choices_from(fetch_manifest(base_url)),
            opening=os.environ.get(VECTOR_ID_ENV) or None,
        )


# ---------------------------------------------------------------------------
# what the app can ask of the engine
# ---------------------------------------------------------------------------


def with_steering(
    request: Mapping[str, Any], vector_id: str, strength: float
) -> dict[str, Any]:
    """``request`` with this page's vector and strength in the steering args.

    The wire shape is copied from ``inspect_providers/steered_provider``:
    ``extra_body={"vllm_xargs": {"steer_vector": "0007", "steer_strength": s}}``.
    Both keys always, never one: the server answers 400 to a strength with no
    vector, and it is right to — that request means to steer and would generate
    from the base model.

    Merged into whatever ``extra_body`` the builder produced rather than
    replacing it — :func:`wire.sampling.build_request` has already put
    ``chat_template_kwargs`` there, which is what turns thinking on, and
    replacing the envelope would turn it off silently.

    Applied *outside* the builder because the builder is the sanctioned way to
    construct a request out of *sampling* parameters, and its allowlist earns
    entries by a measured effect on the token distribution. Steering is not one
    of those: it changes the model, not the sampler, and it is proven by the
    pod's own patch rather than by a row of that matrix.

    ``cache_salt`` travels with them because it is derived from them: it is what
    stops a prompt asked at one strength from being served to the next one out
    of the prefix cache. See :func:`wire.sampling.steering_salt`.
    """
    extra = dict(request.get("extra_body") or {})
    xargs = dict(extra.get(XARGS_FIELD) or {})
    xargs[VECTOR_ARG] = str(vector_id)
    xargs[STRENGTH_ARG] = float(strength)
    extra[XARGS_FIELD] = xargs
    extra[CACHE_SALT_FIELD] = sampling.steering_salt(vector_id, strength)
    return {**request, "extra_body": extra}


class Api:
    """The things the app asks of the engine, as plain methods.

    Kept apart from anything that draws them, so that everything decidable
    without a browser is decidable without one: the shaping of every request and
    answer, the refusals, the labels and the condition stamping are tested by
    calling these.
    """

    def __init__(self, target: Target) -> None:
        self.target = target
        #: The condition this app will send with the next request: which vector,
        #: and how strongly. The engine holds neither — both travel per request —
        #: so this is the only record of them there is.
        #:
        #: A record, not the source. The page's controls are what the condition
        #: is: they are read on every send and written here before the request is
        #: built (:meth:`set_condition`), and they are read here again on every
        #: edit so the labels stay live between sends. What this exists for is
        #: everything that has to know the condition without a control to hand —
        #: the archive tags, the opening state of a conversation, a script
        #: driving :meth:`chat` with no page at all.
        self.vector = target.opening
        self.strength = 0.0
        # Non-blocking, and a held lock is a refusal rather than a wait. Two
        # turns at once would interleave their chunks into one transcript.
        # Queueing would hide that behind a slow page; refusing says which two
        # things collided. Condition writes deliberately do not take it — see
        # `set_condition`.
        self._lock = threading.Lock()

    # -- helpers ------------------------------------------------------------

    @contextmanager
    def _exclusive(self, what: str) -> Iterator[None]:
        if not self._lock.acquire(blocking=False):
            raise Refused(
                f"{what} was requested while this page was already driving the "
                f"engine. A turn holds the page for as long as it is arriving, so "
                f"two of them at once would interleave into one transcript. Wait "
                f"for the turn in flight to finish."
            )
        try:
            yield
        finally:
            self._lock.release()

    def _state(self) -> dict[str, Any]:
        """The app's whole record of the condition: which vector, how strongly.

        There is no sequence number, no mode, no layer, no delta norm and no
        digest here, because nothing in this system reports them.
        """
        return {"vector": self.vector, "strength": self.strength}

    def condition(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """The pair that will be sent, as the page's fields, plus its label.

        Both halves, because both travel with the request and either can be
        moved. What is *not* here is anything the manifest agreed at startup
        beyond the id — the scale, the digest, the layer. Those are properties of
        the arrays the server loaded and hold for the life of both processes; a
        *condition* is what one request carries, and no completion response says
        what was added to it. Reprinting a startup check next to each turn would
        read as a per-turn confirmation of something nothing per turn can
        confirm.
        """
        served = state.get("strength")
        numeric = isinstance(served, (int, float)) and not isinstance(served, bool)
        return {
            "vector": state.get("vector"),
            "strength": served,
            "relative": float(served) if numeric else None,
            "label": format_condition(state),
        }

    # -- the state ----------------------------------------------------------

    def _vector_state(self, vector: Vector) -> dict[str, Any]:
        return {
            "id": vector.id,
            "name": vector.name,
            "layer": vector.layer,
            "norm": vector.norm,
            "relative_per_unit": vector.relative_per_unit,
            # The scale the strength on the wire is multiplied by. Reported by
            # the server, and where this checkout holds the same vector, agreed
            # with it at startup rather than asserted.
            "scale": vector.scale,
            "description": vector.description or vector.describe(),
            "sha256": vector.sha256,
            "label": vector.label,
            "local": vector.local,
        }

    def state(self) -> dict[str, Any]:
        """Everything fixed about this page, plus the condition right now.

        ``vector`` is the one selected and ``vectors`` is everything selectable,
        both from the same records, so the header and the selector cannot come to
        disagree about what is being sent.
        """
        return {
            "vector": self._vector_state(self.target.require(self.vector)),
            "vectors": [self._vector_state(v) for v in self.target.vectors],
            "server": {
                "model": self.target.model,
                "base_url": self.target.base_url,
                "dtype": PROFILE.dtype,
            },
            "range": strength_range(),
            "convention": vectorfmt.STRENGTH_CONVENTION,
            "condition": self.condition(self._state()),
        }

    # -- moving the condition -----------------------------------------------

    def set_condition(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Record the pair this app will send from now on. Both halves at once.

        A local write, and that is the whole change. There is no engine-side
        condition to set: both values travel with each request
        (:func:`with_steering`), so there is nothing to report as honoured and
        the only evidence that either took effect is the reply. Nothing is sent
        from here — as everywhere on this page, only a turn generates.

        One writer for both halves rather than two, because the page reads them
        off its controls together and a half-applied condition is one nothing
        chose. The vector is checked against the served set before anything is
        written, so the page cannot come to hold a vector the server would
        answer 400 to, and normalised, so ``7``, ``"7"`` and ``"0007"`` are one
        condition rather than three the archive would list separately.

        **Deliberately not under** :meth:`_exclusive`. A turn reads the condition
        exactly once, at the top of :meth:`chat` or ``Session.stream_turn``, and
        builds its request from that one read — so a write cannot reach a turn
        that is already in flight, and taking the lock here would do nothing but
        refuse a person editing the controls while a reply arrives. Editing them
        then is ordinary: it is how you line up the next question.
        """
        requested = body.get("vector")
        if requested is None:
            raise BadRequest("no vector was given")
        try:
            canonical = vectorfmt.vector_id(requested)
        except vectorfmt.VectorFormatError as exc:
            raise BadRequest(str(exc)) from exc
        vector = self.target.require(canonical)
        strength = _number(body, "strength")

        self.vector = vector.id
        self.strength = strength
        return {
            "condition": self.condition(self._state()),
            "request": {"vector": vector.id, "strength": strength},
        }

    # -- one turn -----------------------------------------------------------

    def chat(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """One turn, at the condition this app is holding.

        Under the lock, so that two turns cannot interleave. The condition is
        read **once**, into ``state``, and the request is built from that read
        rather than from the record — so an edit made while this is in flight
        moves the next turn and cannot relabel this one. There is nothing to
        stamp before and after.

        Thinking is requested and read leniently, with what was seen reported. A
        strength large enough to stop the model producing an answer is the
        interesting end of the range, and a page that raised there would be
        unable to show the very behaviour the strength causes.

        Answered in one piece, which is why the app does not send a turn through
        here: a reply that takes minutes cannot be watched arriving. This stays as
        the sequence a streamed turn has to reproduce step for step, and the
        payload it has to return field for field.
        """
        messages = _messages(body)
        with self._exclusive("a turn"):
            state = self._state()
            response = self._generate(messages, state)

        choice = _first_choice(response)
        split = sampling.split_message(choice.get("message") or {}, require_thinking=False)
        return {
            "content": split.content,
            "thinking": split.thinking,
            # Reported rather than raised, and reported as a fact about the reply
            # rather than as an error: no thinking at a high strength is a result.
            "thinking_missing": not split.thinking,
            "fields_seen": dict(split.fields_seen),
            "finish_reason": choice.get("finish_reason"),
            "condition": self.condition(state),
        }

    def _generate(
        self, messages: list[dict[str, Any]], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The chat request, through the one builder, with no seed.

        The condition comes in as ``state`` rather than being read off ``self``,
        so that the pair this request carries is the same pair the caller stamped
        the turn with. Reading the record here instead would leave a window
        between the two in which an edit could land, and the reply would be
        labelled with a condition it was not generated under.

        No seed on purpose. Every batch driver in this system seeds so that two
        arms differ by the steering and not by the sampling; here a person is
        asking the same question twice to see the spread, and a fixed seed would
        return the identical reply and read as an intervention with no variance.
        """
        from openai import OpenAI

        request = with_steering(
            sampling.build_request(self.target.model, messages, thinking=True),
            state["vector"],
            state["strength"],
        )
        api = OpenAI(
            base_url=self.target.base_url,
            api_key="EMPTY",
            timeout=CHAT_TIMEOUT_S,
            max_retries=0,
        )
        completion = api.chat.completions.create(**request)
        return completion.model_dump()


# ---------------------------------------------------------------------------
# reading what the app passed in
# ---------------------------------------------------------------------------


class BadRequest(ValueError):
    """The app asked for something this method cannot act on."""


def _number(body: Mapping[str, Any], key: str) -> float:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(f"{key!r} must be a number, got {value!r}")
    return float(value)


def _messages(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The transcript, validated. The app owns the history; this owns its shape."""
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise BadRequest("'messages' must be a non-empty list")
    messages: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise BadRequest(f"messages[{index}] is not an object: {item!r}")
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant", "system"):
            raise BadRequest(f"messages[{index}] has role {role!r}")
        if not isinstance(content, str):
            raise BadRequest(f"messages[{index}] has no text content")
        messages.append({"role": role, "content": content})
    return messages


def _first_choice(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise sampling.ScoringError(f"the server returned no choices: {response!r}")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise sampling.ScoringError(f"the server returned a {type(choice).__name__} choice")
    return dict(choice)


# ---------------------------------------------------------------------------
# failures, classified
# ---------------------------------------------------------------------------

#: Which status each failure the app can produce is classified with, by the
#: dotted name of the exception class.
#:
#: HTTP's numbers, because they are the shared vocabulary for whose fault a
#: failure is and whether it may be retried, and the app acts on them: 409 is
#: waited out and retried, 400 is a defect in the app's own request. Keyed by
#: name rather than by class so the table reads as a list of the things that can
#: go wrong.
#:
STATUS_TABLE: dict[str, int] = {
    "steering_vectors.wire.sampling.UnsupportedParamError": 400,
    "steering_vectors.wire.sampling.MissingThinkingError": 502,
    "steering_vectors.wire.sampling.ScoringError": 502,
    "steering_vectors.vectorfmt.VectorFormatError": 500,
}

#: The app's own three, matched by class because this module's name depends on
#: how it was loaded — it is a file next to the package, not a module inside it.
#: ``NoServerError`` is here rather than in the table above for that reason: it
#: is defined in this file.
LOCAL_STATUS: tuple[tuple[type[Exception], int], ...] = (
    (Refused, 409),
    (BadRequest, 400),
    (NoServerError, 503),
)


def status_for(exc: BaseException) -> int:
    """The status for one failure.

    Unmapped exceptions are 500 and carry their text, because an app that
    swallowed the message would leave a page saying only that something happened.

    409 has exactly one source: this app's own lock.
    """
    for cls, status in LOCAL_STATUS:
        if isinstance(exc, cls):
            return status
    for cls in type(exc).__mro__:
        dotted = f"{cls.__module__}.{cls.__qualname__}"
        if dotted in STATUS_TABLE:
            return STATUS_TABLE[dotted]
    return 500


def error_payload(exc: BaseException) -> dict[str, Any]:
    """What the page is told about a failure: the type, the text, retryability."""
    status = status_for(exc)
    return {
        "error": type(exc).__name__,
        "detail": str(exc),
        "status": status,
        "retryable": status == 409,
    }
