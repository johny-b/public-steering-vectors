"""Read the vector directories in `vectors/`: show one, validate them, index them.

Everything about a vector directory that needs no GPU, no server and no network:

    python scripts/vector_show.py show 0007          # the card
    python scripts/vector_show.py show 0007 --json   # the validated metadata
    python scripts/vector_show.py validate           # every directory
    python scripts/vector_show.py index              # regenerate INDEX.md
    python scripts/vector_show.py index --check      # ... or just check it

`validate` is the one worth running after moving or copying a vector: it checks
that every required file is present, that every file matches the sha256 its own
`meta.json` records, that the metadata is complete and internally consistent,
and that `vector.npy` really is row `layer` of `deltas_all_layers.npy`. That
last check is the only one that can catch a vector derived at one layer and
recorded as another, which otherwise steers the wrong block and still looks
like a result.

`index` rewrites `vectors/INDEX.md` from scratch, header lines included, and
`index --check` fails if the committed file differs from what would be written.
The table is wholly derived — every column is read out of a vector's own
`meta.json` — so nothing in INDEX.md should ever be edited by hand.

Exit codes: 0 ok, 1 something did not validate, 2 a usage error, 3 refused
(asked to check a set of vectors that turned out to be empty, which is a pass
earned by an empty directory rather than a pass).

The optional `demo/` directory is listed but not validated: nothing in this
repository writes a `demo/ladder.json`, so a schema for one here would be an
unowned description of a file that has no producer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

# A plain checkout is enough to run this: the package sits at the repository
# root, so it is importable whether or not `pip install -e .` has been run.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steering_vectors import vectorfmt  # noqa: E402
from steering_vectors.core import canonjson, paths  # noqa: E402
from steering_vectors.vectorfmt import VectorFormatError  # noqa: E402

#: Exit code for a refusal: not "this failed" (1) and not "you typed it wrong"
#: (2), but "doing this would produce a result that cannot be trusted".
REFUSED = 3


# ---------------------------------------------------------------------------
# vectors/INDEX.md
#
# Wholly derived: every column is read out of a vector's own `meta.json`, and
# the file is rewritten from scratch each time.
#
# It exists for one question: *which of these vectors do I want, and what does a
# strength mean on it?* Hence the last three numeric columns, which are the ones
# that are not comparable between vectors unless they are next to each other.
#
# One rule, and it is the reason this is not four lines: **a vector that does
# not load is an error, not an omitted row.** A generated table that skips what
# it cannot read describes a smaller collection than exists on disk, and the
# skipped vector is exactly the one something is wrong with.
# ---------------------------------------------------------------------------

#: How much of a description a row shows. Long descriptions are truncated with an
#: ellipsis, so a reader can tell a shortened one from a short one.
DESCRIPTION_WIDTH = 60


def truncate(text: str, width: int = DESCRIPTION_WIDTH) -> str:
    """`text`, at most `width` characters, marked when it was cut."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[: width - 1].rstrip() + "…"


def row(meta: dict[str, Any], vdir: Path) -> str:
    """One vector's row."""
    demo = "yes" if vectorfmt.has_demo(vdir) else "—"
    verified = vectorfmt.per_prompt_verification(meta)
    checked = "yes" if None not in verified.values() else "—"
    return (
        f"| `{meta['id_str']}` | **{meta['name']}** | {meta['layer']} | "
        f"{meta['n_pos']}/{meta['n_neg']} | {meta['vector_norm']:.3f} | "
        f"{meta['activation_norm_at_layer']:.2f} | "
        f"{meta['vector_norm_over_activation_norm']:.4f} | {checked} | {demo} | "
        f"{meta['created_at']} | {truncate(meta['description'])} |"
    )


def render(vectors_dir: str | Path | None = None) -> str:
    """The whole index, from whatever vector directories exist."""
    root = Path(vectors_dir) if vectors_dir is not None else paths.vectors_dir()
    ids: Sequence[str] = vectorfmt.existing_ids(root) if root.exists() else []
    rows = []
    for vid in ids:
        vdir = root / vid
        try:
            meta = vectorfmt.read_meta(vdir)
        except (VectorFormatError, OSError, ValueError) as exc:
            raise VectorFormatError(
                f"cannot index {vdir}: {exc}\n"
                f"Refusing to write an index that omits it. A generated table that "
                f"skipped unreadable vectors would list fewer vectors than exist, "
                f"and this is the one to look at. Fix or remove the directory."
            ) from exc
        rows.append(row(meta, vdir))
    if not rows:
        rows = [
            "| — | _no vectors_ | | | | | | | | | "
            "add a vector directory under `vectors/` |"
        ]
    return "\n".join(
        [
            "# steering vectors",
            "",
            "One self-describing directory per vector; "
            "`python scripts/vector_show.py show <id>` prints that vector's card.",
            "",
            "Generated by `python scripts/vector_show.py index` — every column is "
            "read from the vector's own `meta.json`, so edits here are lost on the "
            "next run.",
            "",
            "| id | name | layer | n pos/neg | `‖v‖` | act. norm | `‖v‖`/act | "
            "capture checked | demo | created | description |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "`‖v‖` and `act. norm` are both at the vector's designated layer. The "
            "ratio is the",
            "fraction of a typical residual-stream magnitude that `--strength 1.0` "
            "adds, and it",
            "is the only one of these numbers that compares across vectors.",
            "",
            "`capture checked` is whether the vector records a per-prompt count for "
            "its capture",
            "assertions; `—` means no claim is made, not that the check failed.",
            "",
        ]
    )


def write_index(vectors_dir: str | Path | None = None) -> Path:
    """Regenerate `vectors/INDEX.md` and return its path."""
    root = Path(vectors_dir) if vectors_dir is not None else paths.vectors_dir()
    text = render(root)
    paths.ensure_dir(root)
    destination = root / vectorfmt.INDEX_NAME
    canonjson.write_text_atomic(destination, text)
    return destination


# ---------------------------------------------------------------------------
# the subcommands
# ---------------------------------------------------------------------------


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return REFUSED


def _resolve(value: str, root: Path | None) -> Path:
    """A vector id, or a path to a directory. Ids are the normal form."""
    candidate = Path(value)
    if candidate.is_dir():
        return candidate
    base = root if root is not None else paths.vectors_dir()
    return base / vectorfmt.vector_id(value)


def _show(args: argparse.Namespace) -> int:
    vdir = _resolve(args.vector, args.vectors_dir)
    meta = vectorfmt.read_meta(vdir)
    if args.as_json:
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    if args.card:
        # The card proper: the same content as the metadata, written for a
        # person, and part of the vector directory rather than generated here.
        print((vdir / vectorfmt.README_NAME).read_text(encoding="utf-8"), end="")
        return 0
    print(f"{vdir}")
    print(vectorfmt.describe(meta))
    print(
        f"  layer {meta['layer']} of {meta['n_layers']}, "
        f"hidden size {meta['hidden_size']}"
    )
    print(f"  captured through {meta['capture']['engine']}")
    print(
        f"  vector norm / activation norm = "
        f"{meta['vector_norm_over_activation_norm']:.4f}"
    )
    # How the pod will serve this one. Both numbers are derived from the
    # metadata rather than configured anywhere, so printing them here is a
    # report and not a second place they could be set differently.
    try:
        block = vectorfmt.steer_layer(meta)
    except VectorFormatError as exc:
        print(f"  not servable by an output hook: {exc}")
    else:
        print(
            f"  served at block {block}, whose output is the layer "
            f"{meta['layer']} stream, scaled by "
            f"{vectorfmt.steer_scale(meta):.6f}"
        )
    print(f"  demo ladder: {'present' if vectorfmt.has_demo(vdir) else 'absent'}")
    print(f"  card: {vdir / vectorfmt.README_NAME} (--card to print it)")
    return 0


def _validate_one(vdir: Path) -> None:
    vectorfmt.check_files_present(vdir)
    meta = vectorfmt.read_meta(vdir)
    vectorfmt.verify_digests(vdir, meta)
    vector = vectorfmt.load_vector(vdir, meta)
    deltas = vectorfmt.load_deltas(vdir, meta)
    vectorfmt.check_vector_is_delta_row(vector, deltas, int(meta["layer"]))


def _validate(args: argparse.Namespace) -> int:
    root = args.vectors_dir if args.vectors_dir is not None else paths.vectors_dir()
    if args.vectors:
        directories = [_resolve(value, root) for value in args.vectors]
    else:
        directories = [root / vid for vid in vectorfmt.existing_ids(root)]

    if not directories:
        return _refuse(
            f"no vector directories under {root}. Reporting 'all vectors valid' "
            f"having checked none of them would be a pass earned by an empty "
            f"directory."
        )

    problems = 0
    for vdir in directories:
        try:
            _validate_one(vdir)
        except Exception as exc:  # noqa: BLE001 - reported per directory, then counted
            problems += 1
            print(f"{vdir.name}: FAILED\n  {type(exc).__name__}: {exc}")
        else:
            print(f"{vdir.name}: ok")

    print(
        f"{len(directories) - problems} of {len(directories)} vector directories "
        f"valid"
    )
    return 0 if problems == 0 else 1


def _index(args: argparse.Namespace) -> int:
    root = args.vectors_dir if args.vectors_dir is not None else paths.vectors_dir()
    destination = root / vectorfmt.INDEX_NAME
    if args.check:
        wanted = render(root)
        current = (
            destination.read_text(encoding="utf-8") if destination.is_file() else None
        )
        if current == wanted:
            print(f"{destination}: up to date")
            return 0
        print(
            f"{destination}: out of date — regenerate it with "
            f"`python scripts/vector_show.py index`",
            file=sys.stderr,
        )
        return 1
    print(write_index(root))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vector_show.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_vectors_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--vectors-dir",
            type=Path,
            default=None,
            metavar="PATH",
            help="directory of vector directories; default: vectors/ under the repo",
        )

    show = sub.add_parser(
        "show",
        help="print one vector's card",
        description="Read a vector directory's metadata, validate it, and print it.",
    )
    show.add_argument("vector", metavar="<id>", help="vector id, e.g. 0007")
    show.add_argument(
        "--card",
        action="store_true",
        help=f"print the vector's own {vectorfmt.README_NAME} instead of the summary",
    )
    show.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="print the validated metadata as JSON",
    )
    add_vectors_dir(show)
    show.set_defaults(handler=_show)

    validate = sub.add_parser(
        "validate",
        help="check vector directories against the format",
        description=(
            "Check that every required file is present, that every recorded digest "
            "matches the file it names, that the metadata is complete, and that the "
            "stored vector really is the stored delta row for its layer. Exits 1 if "
            "anything does not hold.\n\n"
            "With no id, every vector directory is checked; a run that finds no "
            "directories at all is a refusal rather than a pass."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validate.add_argument(
        "vectors",
        nargs="*",
        metavar="<id>",
        help="vector ids to check; default: every directory found",
    )
    add_vectors_dir(validate)
    validate.set_defaults(handler=_validate)

    index = sub.add_parser(
        "index",
        help="regenerate vectors/INDEX.md",
        description=(
            "Rebuild the index from the vector directories. Derived and pure: the "
            "output is a function of the directories alone, so it is safe to delete "
            "and safe to run twice."
        ),
    )
    index.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the file on disk is not what would be written",
    )
    add_vectors_dir(index)
    index.set_defaults(handler=_index)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except VectorFormatError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
