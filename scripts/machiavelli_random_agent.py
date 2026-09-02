"""Validate the MACHIAVELLI port by playing every game at random.

The benchmark's normalisation coefficients *are* the mean harm counts of a
uniformly random agent, so a uniformly random agent must score ~100 on every
metric. That single number exercises the ChoiceScript interpreter, the
environment's annotation accumulation and the scorer at once: if any of them
drifts, the mean moves off 100. No model and no API key are involved.

Usage::

    python scripts/machiavelli_random_agent.py --seeds 3

Add `--data-dir` (or set `$MACHIAVELLI_DATA_DIR`) to point at an existing copy
of the game data instead of downloading 3.4 GB.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from machiavelli import TEST_GAMES, ensure_game_data, mach_counts, mach_scores
from machiavelli.engine import MachiavelliEnv, Thresholds
from machiavelli.game_data import (
    load_game_annotations,
    load_game_metadata,
    load_game_source,
    load_game_tree,
    load_normalization_coeffs,
)

MAX_TRAJ_LENGTH = 1000
"""The eval's default cap. A random agent averages under 200 decisions."""


def play(args: tuple[Path, str, int]) -> dict[str, object]:
    """Play one game with one seed and score it."""
    data_dir, game, seed = args
    started = time.monotonic()

    env = MachiavelliEnv(
        game,
        load_game_metadata(data_dir, game),
        load_game_tree(data_dir, game),
        load_game_annotations(data_dir, game),
        load_game_source(data_dir, game),
        seed=seed,
    )
    # numpy rather than random.Random, and seeded the way the benchmark's own
    # random agent is, so that this reproduces the published baseline exactly.
    rng = np.random.RandomState(seed)

    steps = 0
    while not env.done and steps < MAX_TRAJ_LENGTH:
        env.step(int(rng.choice(env.info().state.num_choices)))
        steps += 1

    counts = mach_counts(env.trajectory, Thresholds())
    scores = mach_scores(counts, load_normalization_coeffs(data_dir, game))
    return {
        "game": game,
        "seed": seed,
        "steps": steps,
        "completed": env.done,
        "secs": round(time.monotonic() - started, 1),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="seeds per game")
    parser.add_argument("--games", nargs="*", default=list(TEST_GAMES))
    parser.add_argument("--data-dir", default=None, help="where the game data lives")
    parser.add_argument("--out", default=None, help="write per-run results as JSONL")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = ensure_game_data(args.data_dir)
    jobs = [(data_dir, game, seed) for game in args.games for seed in range(args.seeds)]

    results = []
    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(play, jobs):
            results.append(result)
            print(
                f"{result['game']:<45} seed {result['seed']} "
                f"{result['steps']:>4} steps  {result['secs']:>5}s",
                flush=True,
            )

    if args.out:
        with open(args.out, "w") as f:
            for result in sorted(results, key=lambda r: (r["game"], r["seed"])):
                f.write(json.dumps(result) + "\n")

    by_metric: defaultdict[str, list[float]] = defaultdict(list)
    for result in results:
        for metric, value in result["scores"].items():
            by_metric[metric].append(value)

    print(f"\n{len(results)} runs over {len(args.games)} games x {args.seeds} seeds")
    print(f"{'metric':<28}{'mean':>9}{'stderr':>9}")
    for metric in sorted(by_metric):
        values = np.array(by_metric[metric])
        stderr = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        print(f"{metric:<28}{values.mean():>9.1f}{stderr:>9.1f}")

    steps = [r["steps"] for r in results]
    print(f"\nsteps: mean {np.mean(steps):.1f}, max {max(steps)}")
    cut_short = sum(not r["completed"] for r in results)
    print(f"games cut short at {MAX_TRAJ_LENGTH}: {cut_short}")


if __name__ == "__main__":
    main()
