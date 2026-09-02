"""The game loop: hand the agent a scene, hand the engine the answer, repeat."""

from __future__ import annotations

from pathlib import Path

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel, store_as
from pydantic import Field

from .agent import AgentCounters, MachiavelliAgent
from .engine import MachiavelliEnv, Trajectory
from .game_data import (
    load_game_annotations,
    load_game_metadata,
    load_game_source,
    load_game_tree,
)


class MachiavelliState(StoreModel):
    """What the solver leaves behind for the scorer."""

    trajectory: Trajectory | None = Field(default=None)
    """The playthrough. `None` only if the solver raised before finishing."""
    counters: AgentCounters = Field(default_factory=AgentCounters)
    """Instruction-following diagnostics; see `AgentCounters`."""
    completed: bool = Field(default=False)
    """Whether the story reached an ending rather than running out of steps."""
    steps: int = Field(default=0)
    """Decisions actually taken."""


@solver
def machiavelli_solver(
    data_dir: Path,
    max_traj_length: int = 1000,
    max_parse_retries: int = 2,
) -> Solver:
    """Play one game to its ending, or until `max_traj_length` decisions.

    Args:
        data_dir: The populated game-data directory; see `game_data`.
        max_traj_length: Decisions after which to stop and score what happened.
            The benchmark's own cap. Most games end well inside it -- a random
            agent averages under 200 decisions -- but a few can loop.
        max_parse_retries: Regenerations allowed for a reply that contains no
            usable choice, before falling back to a random one.

    The epoch number is the seed: epoch 1 of a game is a different playthrough
    from epoch 2 wherever the story rolls dice, and the same across runs.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        game = state.metadata["game"]
        seed = state.epoch

        env = MachiavelliEnv(
            game,
            load_game_metadata(data_dir, game),
            load_game_tree(data_dir, game),
            load_game_annotations(data_dir, game),
            load_game_source(data_dir, game),
            seed=seed,
        )
        agent = MachiavelliAgent(seed=seed, max_parse_retries=max_parse_retries)

        steps = 0
        while not env.done and steps < max_traj_length:
            action = await agent.act(env.obs, env.info())
            env.step(action)
            steps += 1

        store = store_as(MachiavelliState)
        store.trajectory = env.trajectory
        store.counters = agent.counters
        store.completed = env.done
        store.steps = steps

        # Also on the sample so it shows up next to the scores without having
        # to dig into the store.
        state.metadata.update(agent.counters.model_dump())
        state.metadata["steps"] = steps
        state.metadata["completed"] = env.done

        return state

    return solve
