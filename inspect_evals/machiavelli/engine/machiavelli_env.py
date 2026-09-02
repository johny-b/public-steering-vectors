"""The environment: one game, played one decision at a time.

Adapted from
https://github.com/aypan17/machiavelli/blob/main/machiavelli/game/machiavelli_env.py.
`MachiavelliEnv` wraps a `Player` and turns it into a step/observe loop: it
hands out the narration and the choices, applies an action, and accumulates the
harm annotations of the scenes the story passed through into a `Trajectory` for
the scorer.

Changes from upstream:

* Randomness is per-environment. Upstream re-seeded the global `random` and
  `numpy.random` in `reset()`, which makes two games running concurrently in one
  process perturb each other. The seed now drives a `random.Random` owned by
  this environment, which draws the same sequence, so a trajectory is unchanged
  but is also reproducible under concurrency.
* The trajectory stores what the scorer actually reads -- the action, the score
  and the summed annotations of each step -- rather than the whole observation
  and info dict. Upstream's `info` repeated the game's full achievement list on
  every step, roughly 17 kB per decision, which for a 1000-step game is 18 MB of
  eval log per sample. The narration is not duplicated either: Inspect already
  records every prompt and completion in the sample transcript.
* The variable-based annotations (`_get_variables`,
  `_get_variable_annotations`) are gone. They were recomputed on every step and
  no MACHIAVELLI metric reads them.
* `render`, `close`, `_get_state`/`_set_state` and the gym-style `metadata` are
  gone; nothing in an Inspect eval calls them.

Upstream's off-by-one on the final step (see `step`) is deliberately preserved:
the published normalisation coefficients were measured with it.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from .player import Player
from .types import (
    Achievement,
    GameAnnotations,
    GameInfo,
    GameMetadata,
    GameSource,
    GameState,
    GameTreeNode,
    MachiavelliEngineError,
    SceneAnnotations,
)


class TrajectoryStep(BaseModel):
    """One decision point, reduced to what the MACHIAVELLI metrics need."""

    action: int
    """The choice taken, or -1 for the terminal state, which offers none."""
    points: int
    """Achievement points earned so far."""
    annotations: SceneAnnotations | None
    """Summed annotations of the scenes traversed to reach this decision.

    `None` when none of those scenes were annotated. The values are sums over
    scenes rather than the per-scene fractions of annotators, which is why the
    scorer's thresholds are compared against them without averaging.
    """


class Trajectory(BaseModel):
    """A complete (or truncated) playthrough, as the scorer sees it."""

    game: str
    max_points: int
    """Points available if every achievement in the game were unlocked."""
    steps: list[TrajectoryStep] = Field(default_factory=list)


def _add_annotations(left: Any, right: Any) -> Any:
    """Add two `SceneAnnotations.model_dump()` results together, leaf by leaf.

    >>> _add_annotations({"a": 1, "b": {"c": 2}}, {"a": 4, "b": {"c": 5}})
    {'a': 5, 'b': {'c': 7}}
    """
    if isinstance(left, dict):
        return {k: _add_annotations(left[k], right[k]) for k in left}
    return left + right


class MachiavelliEnv:
    """One playthrough of one game.

    Args:
        game: The game's identifier in the data directory, e.g. `avatar-of-the-wolf`.
        game_metadata: The game's `game_metadata.json`.
        game_tree: The game's scene graph, keyed by node name.
        game_annotations: The game's per-scene harm annotations.
        game_source: The game's parsed ChoiceScript.
        seed: Seed for the story's own randomness (`*rand`, `*goto_random_scene`).
            Re-applied on every `reset`, so the same seed replays the same game.
    """

    def __init__(
        self,
        game: str,
        game_metadata: GameMetadata,
        game_tree: Mapping[str, GameTreeNode],
        game_annotations: GameAnnotations,
        game_source: GameSource,
        seed: int = 0,
    ) -> None:
        self.game = game
        self.seed = seed
        self.title = game_metadata["name"]
        self.short_summary = game_metadata["short_summary"]
        self.player_character = game_metadata["player_character"]
        self.node_to_annotations = game_annotations.scenes

        self.rng = random.Random()
        self.player = Player(game_tree, game_source, rng=self.rng)
        # The engine only declares the game's achievements once it has run the
        # startup scene, so play up to the first choice before reading them.
        self.player.skip_to_next_choice()

        self.achievements: dict[str, Achievement] = {}
        for name, (
            _visible,
            points,
            title,
            locked,
            unlocked,
        ) in self.player.get_achievements().items():
            description = locked if locked == unlocked else f"{locked} / {unlocked}"
            self.achievements[name] = Achievement(
                description=f"{title}: {description}", points=points
            )
        self.max_points = sum(a["points"] for a in self.achievements.values())

        self.reset()

    def reset(self) -> str:
        """Restart the game and return the opening narration."""
        self.rng.seed(self.seed)
        self.player.reset()
        self.curr_node_ids = _flatten(self.player.skip_to_next_choice())
        self.obs = self.player.flush_print_buffer()
        self.trajectory = Trajectory(game=self.game, max_points=self.max_points)
        self._points = self.points
        self._annotations = self._step_annotations()
        return self.obs

    @property
    def done(self) -> bool:
        """Whether the story has reached an ending."""
        return self.player.done

    @property
    def points(self) -> int:
        """Achievement points earned so far."""
        return sum(
            self.achievements[a]["points"]
            for a in self.player.unlocked_achievements
            if a in self.achievements
        )

    def info(self) -> GameInfo:
        """Everything the agent is shown besides the narration itself."""
        return GameInfo(
            title=self.title,
            short_summary=self.short_summary,
            player_character=self.player_character,
            achievements=self.achievements,
            max_points=self.max_points,
            state=GameState(
                num_choices=self.player.get_action_space(),
                choice_texts=self.player.get_choice_texts(),
                current_nodes=list(self.curr_node_ids),
                unlocked_achievements=[
                    a
                    for a in self.achievements
                    if a in self.player.unlocked_achievements
                ],
                points=self.points,
            ),
        )

    def step(self, action: int) -> str:
        """Take a choice, run the story on to the next one, and return the narration."""
        if self.done:
            raise MachiavelliEngineError("step() called on a finished game")
        if not 0 <= action < self.player.get_action_space():
            raise ValueError(
                f"invalid action {action}: "
                f"expected 0 <= action < {self.player.get_action_space()}"
            )

        # Recorded before stepping, so that the annotations attributed to a
        # decision are those of the scenes the player read before making it.
        self.trajectory.steps.append(
            TrajectoryStep(
                action=action, points=self._points, annotations=self._annotations
            )
        )

        self.player.play(action)
        self.curr_node_ids = _flatten(self.player.skip_to_next_choice())
        self.obs = self.player.flush_print_buffer()

        if self.player.done:
            # Upstream records the terminal state before refreshing the info it
            # records, so the last entry repeats the previous step's score and
            # annotations instead of the final scene's. The published
            # normalisation coefficients were measured against a random agent
            # with this behaviour, so correcting it here would shift every
            # reported score by a percent or so relative to the paper.
            self.trajectory.steps.append(
                TrajectoryStep(
                    action=-1, points=self._points, annotations=self._annotations
                )
            )

        self._points = self.points
        self._annotations = self._step_annotations()
        return self.obs

    def _step_annotations(self) -> SceneAnnotations | None:
        """Sum the annotations of every scene traversed since the last decision.

        A single step can run through several scenes, and a scene can be
        annotated or not; the sum is over the annotated ones, and is `None` if
        there were none.
        """
        total: dict[str, Any] | None = None
        for node in self.curr_node_ids:
            annotations = self.node_to_annotations.get(node)
            if annotations is None:
                continue
            dumped = annotations.model_dump()
            total = dumped if total is None else _add_annotations(total, dumped)
        return None if total is None else SceneAnnotations.model_validate(total)


def _flatten(nodes: list[list[str]]) -> list[str]:
    """Flatten the per-line node lists `Player.skip_to_next_choice` returns."""
    return [node for node_list in nodes for node in node_list]
