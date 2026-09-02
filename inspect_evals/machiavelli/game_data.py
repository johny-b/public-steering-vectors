"""Getting the MACHIAVELLI game data onto disk, and reading it back typed.

The benchmark's 134 games are not redistributable as plain text -- the original
authors ship them as a password-protected zip so that the ChoiceScript sources
stay out of web crawls and training sets. This module downloads that archive
from HuggingFace once, unpacks it into a cache directory, and then loads
individual games out of it.

A cache directory holds the data directly, so it looks like this::

    <data_dir>/
        game_metadata.json          # every game's title, summary, achievements
        normalization_coeffs.json   # every game's random-agent baseline
        source/<game>.json          # parsed ChoiceScript
        trees/<game>_game_tree.json # scene graph
        annotations_clean/<game>.json

It is about 3.4 GB unpacked, which is why the location is configurable: point
`data_dir` (or `$MACHIAVELLI_DATA_DIR`) at a shared volume rather than
re-downloading it per machine.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from functools import cache
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from pydantic import TypeAdapter

from .engine import (
    GameAnnotations,
    GameMetadata,
    GameSource,
    GameTreeNode,
    NormalizationCoeffs,
)

GAME_DATA_REPO = "kobyjl/machiavelli_game_data"
"""HuggingFace dataset holding the archive the original benchmark distributes."""

GAME_DATA_ARCHIVE = "game_data.zip"

GAME_DATA_PASSWORD = b"machiavelli"
"""The archive password. The benchmark publishes it;
it keeps crawlers out, not people."""

DEFAULT_DATA_DIR = Path("~/.cache/machiavelli")
"""Where the game data lives if neither caller nor environment says otherwise."""


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Decide where the game data should live.

    Args:
        data_dir: An explicit location, or `None` to fall back to
            `$MACHIAVELLI_DATA_DIR` and then to `~/.cache/machiavelli`.
    """
    if data_dir is None:
        data_dir = os.environ.get("MACHIAVELLI_DATA_DIR") or DEFAULT_DATA_DIR
    return Path(data_dir).expanduser().resolve()


def ensure_game_data(data_dir: str | Path | None = None) -> Path:
    """Return a populated game-data directory, downloading the archive if needed.

    Call this once, before any samples run: it is not safe to call concurrently
    from several processes pointed at the same directory.

    Args:
        data_dir: Where to keep the data; see `resolve_data_dir`.
    """
    resolved = resolve_data_dir(data_dir)
    if (resolved / "game_metadata.json").exists():
        return resolved

    latest_revision = HfApi().dataset_info(GAME_DATA_REPO).sha
    archive_path = hf_hub_download(
        repo_id=GAME_DATA_REPO,
        filename=GAME_DATA_ARCHIVE,
        repo_type="dataset",
        revision=latest_revision,
    )

    # Unpack beside the destination and move into place, so an interrupted
    # download never leaves a half-populated cache that later runs would trust.
    staging_dir = resolved.parent / f".{resolved.name}.partial"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(path=staging_dir, pwd=GAME_DATA_PASSWORD)
        shutil.rmtree(resolved, ignore_errors=True)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        (staging_dir / "game_data").rename(resolved)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return resolved


_GAME_METADATA = TypeAdapter(GameMetadata)
_GAME_TREE = TypeAdapter(dict[str, GameTreeNode])
_NORMALIZATION_COEFFS = TypeAdapter(NormalizationCoeffs)


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


@cache
def _load_json_cached(path: Path) -> Any:
    """For the two files that describe every game at once, rather than one game."""
    return _load_json(path)


def load_game_metadata(data_dir: Path, game: str) -> GameMetadata:
    """The game's title, summary, player character and achievement list."""
    all_metadata = _load_json_cached(data_dir / "game_metadata.json")
    return _GAME_METADATA.validate_python(all_metadata[game])


def load_normalization_coeffs(data_dir: Path, game: str) -> NormalizationCoeffs:
    """The random agent's mean and standard deviation per metric, for this game.

    Scores are reported relative to these, so a random agent scores 100
    everywhere by construction.
    """
    all_coeffs = _load_json_cached(data_dir / "normalization_coeffs.json")
    return _NORMALIZATION_COEFFS.validate_python(all_coeffs[game])


def load_game_tree(data_dir: Path, game: str) -> dict[str, GameTreeNode]:
    """The game's scene graph, keyed by node name."""
    return _GAME_TREE.validate_python(
        _load_json(data_dir / "trees" / f"{game}_game_tree.json")
    )


def load_game_annotations(data_dir: Path, game: str) -> GameAnnotations:
    """The per-scene harm annotations the scorer counts."""
    return GameAnnotations.model_validate(
        _load_json(data_dir / "annotations_clean" / f"{game}.json")
    )


def load_game_source(data_dir: Path, game: str) -> GameSource:
    """The game's parsed ChoiceScript, ready for the interpreter."""
    source = GameSource.model_validate(_load_json(data_dir / "source" / f"{game}.json"))
    # Append the implicit end-of-scene transition ChoiceScript performs when
    # execution falls off the end of a scene, plus a buffer line so the
    # interpreter's lines[idx + 1] lookahead stays in range on the final command.
    for scene, lines in source.scenes.items():
        lines.append(("*finish" if scene != source.scene_list[-1] else "*ending", 0))
        lines.append(("", 0))
    return source
