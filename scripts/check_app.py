"""What the app can be checked on without a GPU behind it.

Everything here runs against a stub server that answers ``/v1/models`` and
``/steering/vectors`` and never asks for a completion, so it is exactly the part
of the app that does not need the engine: the range, the shape of the request a
condition goes out in, the startup check that the server and this checkout agree
about the vectors they both hold, what moving either half of the condition does
to a conversation, and the layering the app promises to keep.

The manifest stub is built from the real ``vectors/*/meta.json`` through the same
functions the pod uses, so "the app agrees with the server" is checked against
the derivation both sides run rather than against a constant copied into this
file.

Run it with ``python scripts/check_app.py``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"

#: The vectors the stub server offers that this checkout also holds. Described
#: from the real directories rather than written here: a literal digest in this
#: file would be a second copy of the thing the check exists to compare. Two of
#: them, because one vector cannot exercise a selector.
LOCAL_IDS = ("0007", "0008")

#: A vector the stub server offers that this checkout does *not* hold — the other
#: case the app must handle. It is not a mismatch and must not be refused: a
#: checkout older than the pod's vectors directory is ordinary, and the page
#: describes such a vector from the manifest alone and says that is where the
#: figures came from.
REMOTE_ID = "0099"

#: Which vector the page opens on. First in the manifest, with STEER_VECTOR_ID
#: unset.
VECTOR_ID = LOCAL_IDS[0]

#: The app is a client of the server's HTTP API and nothing else: it must not
#: reach into the pod's patch, the evals or the inspect providers.
FORBIDDEN_IMPORTS = ("vllm_steering", "inspect_evals", "steered_provider")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{' — ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def stub_manifest() -> dict:
    """The manifest a correctly configured server would serve for these vectors.

    Built out of the same ``meta.json`` files and the same two derivations the
    pod would build it from, so the startup check this stands in for is
    exercised rather than bypassed. One synthetic entry is appended for
    :data:`REMOTE_ID`, which no directory here backs.
    """
    from steering_vectors import vectorfmt

    metas = [vectorfmt.read_meta(vectorfmt.dir_for(vid)) for vid in LOCAL_IDS]
    entries = [
        {
            "id": meta["id_str"],
            "name": meta["name"],
            "description": meta["description"],
            "layer": meta["layer"],
            "block": vectorfmt.steer_layer(meta),
            "scale": vectorfmt.steer_scale(meta),
            "vector_norm": meta["vector_norm"],
            "relative_per_unit": meta["vector_norm_over_activation_norm"],
            "sha256": meta["vector_npy_sha256"],
        }
        for meta in metas
    ]
    entries.append(
        {
            "id": REMOTE_ID,
            "name": "not-in-this-checkout",
            "description": "a vector the server holds and this tree does not",
            "layer": metas[0]["layer"],
            "block": vectorfmt.steer_layer(metas[0]),
            "scale": 5.0,
            "vector_norm": 6.25,
            "relative_per_unit": 0.2,
            "sha256": "a" * 64,
        }
    )
    return {
        "model": metas[0]["model"],
        "block": vectorfmt.steer_layer(metas[0]),
        "digest": "0" * 64,
        "arg": {
            "field": "vllm_xargs",
            "vector": "steer_vector",
            "strength": "steer_strength",
        },
        "vectors": entries,
    }


class Models(BaseHTTPRequestHandler):
    """Enough of the server for ``list_models`` and the vector check."""

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's name
        path = self.path.rstrip("/")
        if path == "/v1/models":
            payload = {"object": "list", "data": [{"id": MODEL}]}
        elif path == "/steering/vectors":
            payload = stub_manifest()
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D102 - silence the access log
        pass


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    os.environ["STEER_BASE_URL"] = base_url
    os.environ.pop("STEER_MODEL", None)
    os.environ.pop("STEER_VECTOR_ID", None)

    import engine as enginelib
    import session as sessionlib

    # -- the range ----------------------------------------------------------
    rng = enginelib.strength_range()
    check(
        "strength_range is ±1.0",
        rng == {"min": -1.0, "max": 1.0},
        repr(rng),
    )

    # -- the target, resolved from the environment --------------------------
    target = enginelib.Target.resolve()
    check(
        "Target.resolve reads the base URL, the served model and the whole list",
        (target.base_url, target.model, [v.id for v in target.vectors])
        == (base_url, MODEL, [*LOCAL_IDS, REMOTE_ID]),
        f"{target.base_url} {target.model} {[v.id for v in target.vectors]}",
    )
    check(
        "with STEER_VECTOR_ID unset the page opens on the first vector listed",
        target.opening == VECTOR_ID,
        target.opening,
    )
    check(
        "scale is 1 / vector_norm_over_activation_norm",
        abs(target.require("0007").scale - 7.320113853761506) < 1e-12,
        repr(target.require("0007").scale),
    )
    check(
        "the vectors this checkout holds are marked as cross-checked, the other not",
        [v.local for v in target.vectors] == [True] * len(LOCAL_IDS) + [False],
        repr([(v.id, v.local) for v in target.vectors]),
    )

    # -- the vectors are agreed with the server, not asserted ---------------
    for name, mangle in (
        (
            "a different array under an id this checkout holds",
            lambda m: m["vectors"][0].update(sha256="f" * 64),
        ),
        (
            "a different scale for a vector this checkout holds",
            lambda m: m["vectors"][0].update(scale=1.0),
        ),
        ("a manifest with no vectors at all", lambda m: m.update(vectors=[])),
    ):
        manifest = stub_manifest()
        mangle(manifest)
        try:
            enginelib.choices_from(manifest)
        except Exception:  # noqa: BLE001 - any refusal is the pass here
            refused = True
        else:
            refused = False
        check(f"choices_from refuses {name}", refused)

    try:
        enginelib.Target(
            base_url=base_url,
            model=MODEL,
            vectors=enginelib.choices_from(stub_manifest()),
            opening="0123",
        )
    except Exception:  # noqa: BLE001
        refused = True
    else:
        refused = False
    check("Target refuses an opening vector the server does not serve", refused)

    api = enginelib.Api(target)
    session = sessionlib.Session(api)

    # -- the condition on the wire ------------------------------------------
    from steering_vectors.wire import sampling

    ok = True
    detail = ""
    for s in (0.0, 0.3, -0.55, 1.0):
        request = enginelib.with_steering(
            sampling.build_request(target.model, [{"role": "user", "content": "hi"}],
                                   thinking=True),
            VECTOR_ID,
            s,
        )
        xargs = request["extra_body"]["vllm_xargs"]
        if xargs.get("steer_strength") != s:
            ok, detail = False, f"{s} -> {xargs.get('steer_strength')!r}"
            break
        if xargs.get("steer_vector") != VECTOR_ID:
            ok, detail = False, f"vector {xargs.get('steer_vector')!r}"
            break
        if not request["extra_body"].get("chat_template_kwargs"):
            ok, detail = False, f"chat_template_kwargs lost at strength {s}"
            break
    check(
        "build_request + with_steering carries the vector and the strength",
        ok,
        detail,
    )

    # The same, through the path a real send takes: the session builds the
    # request out of the api's own record, not out of a widget.
    sent: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            sent.update(kwargs)
            return iter(())

    class FakeClient:
        chat = type("C", (), {"completions": FakeCompletions()})()

    import openai

    real = openai.OpenAI
    openai.OpenAI = lambda **kwargs: FakeClient()
    try:
        session.apply({"vector": VECTOR_ID, "strength": 0.4})
        list(session._stream_completion([{"role": "user", "content": "hi"}], {}))
    finally:
        openai.OpenAI = real
    check(
        "Session._stream_completion sends the strength the app is holding",
        sent.get("extra_body", {}).get("vllm_xargs", {}).get("steer_strength") == 0.4,
        repr(sent.get("extra_body")),
    )
    check(
        "Session._stream_completion names the vector the app is holding",
        sent.get("extra_body", {}).get("vllm_xargs", {}).get("steer_vector")
        == VECTOR_ID,
        repr(sent.get("extra_body")),
    )
    check(
        "Session._stream_completion streams",
        sent.get("stream") is True,
        repr(sent.get("stream")),
    )
    # The condition is carried in, not read off the api a second time: that is
    # what lets the controls stay live while a reply is arriving.
    sent.clear()
    openai.OpenAI = lambda **kwargs: FakeClient()
    try:
        list(
            session._stream_completion(
                [{"role": "user", "content": "hi"}],
                {},
                {"vector": "0008", "strength": -0.25},
            )
        )
    finally:
        openai.OpenAI = real
    xargs = sent.get("extra_body", {}).get("vllm_xargs", {})
    check(
        "a turn's request is built from the condition it was stamped with",
        (xargs.get("steer_vector"), xargs.get("steer_strength")) == ("0008", -0.25)
        and (api.vector, api.strength) == (VECTOR_ID, 0.4),
        f"{xargs!r} while the record holds {api.vector} {api.strength}",
    )

    # -- reading a typed strength -------------------------------------------
    reads = {
        "0.3": 0.3,
        "0,3": 0.3,
        "  -0,25 ": -0.25,
        "+1": 1.0,
        "0": 0.0,
        0.5: 0.5,
    }
    wrong = [
        f"{raw!r} -> {sessionlib.parse_strength(raw, -1.0, 1.0)}"
        for raw, want in reads.items()
        if sessionlib.parse_strength(raw, -1.0, 1.0) != (want, "")
    ]
    check("a comma reads as a decimal point, like a dot", not wrong, "; ".join(wrong))

    unusable = ["1,000.5", "0,3,4", "", "  ", None, "abc", "inf", "nan", "2", "-1.5"]
    slipped = [
        repr(raw)
        for raw in unusable
        if sessionlib.parse_strength(raw, -1.0, 1.0)[0] is not None
    ]
    check(
        "ambiguous, unreadable and out-of-range strengths are refused with a reason",
        not slipped
        and all(sessionlib.parse_strength(r, -1.0, 1.0)[1] for r in unusable),
        "; ".join(slipped),
    )
    check(
        "a typed strength is never snapped to a step",
        sessionlib.parse_strength("0.354", -1.0, 1.0) == (0.354, ""),
        repr(sessionlib.parse_strength("0.354", -1.0, 1.0)),
    )

    # -- selecting a vector -------------------------------------------------
    check(
        "state lists every served vector for the selector",
        [v["id"] for v in api.state()["vectors"]] == [*LOCAL_IDS, REMOTE_ID],
        repr([v["id"] for v in api.state()["vectors"]]),
    )
    cleaned, refusal = session.clean_condition({"vector": "8", "strength": "0,15"})
    check(
        "clean_condition normalises the id and reads the strength",
        cleaned == {"vector": "0008", "strength": 0.15} and not refusal,
        f"{cleaned!r} {refusal}",
    )
    check(
        "a vector the server does not serve refuses the whole condition",
        session.clean_condition({"vector": "0123", "strength": "0"})[1].startswith(
            "vector:"
        ),
        session.clean_condition({"vector": "0123", "strength": "0"})[1],
    )
    check(
        "an unreadable strength refuses the whole condition",
        session.clean_condition({"vector": VECTOR_ID, "strength": "0.3.1"})[
            1
        ].startswith("strength:"),
        session.clean_condition({"vector": VECTOR_ID, "strength": "0.3.1"})[1],
    )

    # -- applying is immediate, and archives once per decision ---------------
    # Turns put on by hand, because there is no engine here to produce one.
    def converse(vector: str, strength: float, count: int = 1) -> None:
        for i in range(count):
            session.active.turns.append(
                sessionlib.Turn(
                    user=f"q{i}", assistant=f"a{i}", condition="…",
                    vector=vector, strength=strength,
                )
            )

    converse(VECTOR_ID, 0.4, 2)
    outcomes = [
        session.apply({"vector": VECTOR_ID, "strength": typed})
        for typed in ("0", "0,3", "0.35", "0.354")
    ]
    check(
        "every edit is applied at once, both halves together",
        (api.vector, api.strength) == (VECTOR_ID, 0.354),
        f"{api.vector} at {api.strength}",
    )
    # The point of archiving from the keystroke: four of them, one archive. The
    # first that moves the condition empties the conversation, and the rest find
    # nothing left to file.
    check(
        "typing a strength out digit by digit archives once, not once per digit",
        len(session.archive) == 1
        and [o.archived for o in outcomes] == [True, False, False, False],
        f"{len(session.archive)} archived from {[o.archived for o in outcomes]}",
    )
    check(
        "under the pair that produced it, not the one being typed",
        session.archive[0].tag() == f"{VECTOR_ID} at strength +0.4"
        and len(session.archive[0].turns) == 2,
        session.archive[0].tag(),
    )
    check(
        "and its opening question comes back to be asked again",
        outcomes[0].prompt == "q0" and session.active.empty,
        repr(outcomes[0].prompt),
    )
    refused = session.apply({"vector": "0123", "strength": "0.1"})
    check(
        "an unusable edit is refused and neither half is written",
        not refused.ok and (api.vector, api.strength) == (VECTOR_ID, 0.354),
        f"{refused.message} (holding {api.vector} at {api.strength})",
    )

    # -- nothing is archived out from under a reply that is arriving ---------
    converse("0008", 0.1, 1)
    session.active.turns[-1].vector = VECTOR_ID
    session.active.turns[-1].strength = 0.354
    session.streaming = True
    mid = session.apply({"vector": "0008", "strength": "0.9"})
    session.streaming = False
    check(
        "an edit during a reply applies but archives nothing",
        mid.ok and not mid.archived and len(session.archive) == 1
        and (api.vector, api.strength) == ("0008", 0.9),
        f"{len(session.archive)} archived, holding {api.vector} at {api.strength}",
    )
    check(
        "and the send that follows archives it instead",
        session.split_if_moved({"vector": "0008", "strength": 0.9})
        and len(session.archive) == 2
        and session.archive[1].tag() == f"{VECTOR_ID} at strength +0.354",
        f"{len(session.archive)} archived",
    )

    # -- the split reads the pair it is handed -------------------------------
    converse("0008", 0.9, 1)
    check(
        "a turn under the condition that produced the transcript joins it",
        not session.split_if_moved({"vector": "0008", "strength": 0.9})
        and len(session.archive) == 2,
        f"{len(session.archive)} archived",
    )
    check(
        "and the split reads the pair it is handed, not the record",
        session.split_if_moved({"vector": "0008", "strength": 0.2})
        and len(session.archive) == 3,
        f"{len(session.archive)} archived",
    )

    # -- an unusable box refuses the send rather than falling back -----------
    session.apply({"vector": VECTOR_ID, "strength": "0.4"})
    held = (api.vector, api.strength)
    refusals = [
        list(session.send_stream("hello", {"vector": VECTOR_ID, "strength": bad}))[-1]
        for bad in ("", "1,000.5", "9")
    ]
    check(
        "a strength the box cannot read stops the send, holding the last good pair",
        all(
            p.outcome and not p.outcome.ok and "strength" in p.outcome.message
            for p in refusals
        )
        and (api.vector, api.strength) == held,
        "; ".join(str(p.outcome and p.outcome.message) for p in refusals),
    )
    session.archive.clear()
    session.active = session.open_conversation()

    # -- the label does not double-convert ----------------------------------
    label = enginelib.format_condition({"vector": "0007", "strength": 0.3})
    check(
        "format_condition reports the sent value, not a re-scaled one",
        "+0.3" in label and "+30.0%" in label and "0007" in label,
        label,
    )

    # -- the layering -------------------------------------------------------
    import app as applib

    hits = []
    for path in sorted((ROOT / "app").glob("*.py")) + [
        ROOT / "steering_vectors" / "wire" / "sampling.py",
        ROOT / "steering_vectors" / "wire" / "__init__.py",
    ]:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if any(f"{pkg}" in stripped for pkg in FORBIDDEN_IMPORTS):
                hits.append(f"{path.name}:{lineno} {stripped}")
    check(
        "no app module imports the pod, the evals or the inspect providers",
        not hits,
        "; ".join(hits),
    )

    loaded = sorted(
        m
        for m in sys.modules
        if any(m == pkg or m.startswith(pkg + ".") for pkg in FORBIDDEN_IMPORTS)
    )
    check("none of them was imported at runtime either", not loaded, str(loaded))

    # Still absent, and still for the same reason: reading the intervention back
    # would need the engine's activations, which no completion response carries.
    for name in ("readout", "recheck", "_readout_payload"):
        check(
            f"engine has no {name}",
            not hasattr(enginelib.Api, name) and not hasattr(enginelib.Target, name),
        )
    check("app.py has no probing handler", not hasattr(applib, "on_probe"))
    check(
        "app.py has one handler for both condition controls",
        hasattr(applib, "on_condition"),
    )
    # The set button and everything that existed to serve it. What is on screen
    # is what is sent, so there is nothing left to press to make that true.
    gone = [
        name
        for name in ("on_set", "on_vector", "steps_note", "condition_line")
        if hasattr(applib, name)
    ] + [
        name
        for name in ("change_strength", "change_vector", "set_strength_typed")
        if hasattr(sessionlib.Session, name)
    ]
    check("no set-button path is left to reach the engine by", not gone, str(gone))

    # -- the page builds ----------------------------------------------------
    applib.SESSION = session
    demo = applib.build(session)
    check("the Blocks page builds", demo is not None)
    inputs = {
        component
        for dependency in demo.fns.values()
        for component in dependency.inputs
    }
    outputs = {
        component
        for dependency in demo.fns.values()
        for component in dependency.outputs
    }
    strength, vector = None, None
    for component in demo.blocks.values():
        label = getattr(component, "label", None)
        if label == "strength":
            strength = component
        elif label == "vector":
            vector = component
    check(
        "the strength is a text box, so a typed comma reaches the parser",
        isinstance(strength, applib.gr.Textbox),
        type(strength).__name__,
    )
    check(
        "both condition controls feed the handlers and neither is written back",
        strength in inputs
        and vector in inputs
        and strength not in outputs
        and vector not in outputs,
        f"strength in={strength in inputs} out={strength in outputs}, "
        f"vector in={vector in inputs} out={vector in outputs}",
    )

    print()
    print(f"{'FAILED' if failures else 'OK'} — {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
