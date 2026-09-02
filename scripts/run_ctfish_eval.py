"""Run the ctfish agent loop and save the logs; the judges are a separate pass.

    python scripts/run_ctfish_eval.py --model openai/gpt-4o-mini \
        --variant baseline --epochs 10 --log-dir logs/ctfish

    python scripts/run_ctfish_eval.py --model steered/<served-model-name> \
        --steer-vector 0001 --steer-strength 0.3 \
        --variant baseline dontlook --epochs 20 --log-dir logs/ctfish

Several variants in one invocation become one sample each in a single eval, so
they share a server, a queue and a wall-clock window; a drift in the serving
stack then lands on all of them or none.

The LLM judges are off by default. They are a second pass over the saved logs
(`scripts/score_ctfish.py`), which keeps two things apart: the agent pass costs
sandbox and GPU time and needs no judge credentials, and a judge outage cannot
waste work that has already been paid for. The deterministic scorers
(`won_game`, `moves_submitted`, `engine_assisted_moves`,
`truncated_generations`) run in both passes, with no model in the loop.

Requires Docker, and a reachable server for whichever model is named.
"""

from __future__ import annotations

import argparse
import sys

from ctfish import ctfish
from inspect_ai import eval


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ctfish agent loop over one or more prompt variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Inspect model name. Note that upstream's `model_config`/"
        "`pre_invoke_message` match on substrings of it (`o1`, `o3`, `deepseek`), "
        "so a name containing one of those silently changes the agent's prompt.",
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=["baseline"],
        help="Task prompt variant(s); see ctfish.prompts.TASK_PROMPTS.",
    )
    parser.add_argument(
        "--steer-vector", default=None, help="Steering vector id, for a steered model."
    )
    parser.add_argument(
        "--steer-strength",
        type=float,
        default=None,
        help="Steering strength. 0.0 is not the same as omitting both arguments: "
        "the vector is still applied, at zero, so a 0.0 run is comparable with "
        "steered ones rather than merely similar to them.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Runs per variant.")
    parser.add_argument(
        "--max-sandboxes", type=int, default=5, help="Concurrent containers."
    )
    parser.add_argument("--log-dir", default="logs/ctfish", help="Eval log directory.")
    parser.add_argument("--max-steps", type=int, default=37)
    parser.add_argument("--command-timeout", type=int, default=30)
    parser.add_argument(
        "--generate-config",
        default=None,
        help="Agent sampling preset, e.g. 'qwen3-thinking'. None keeps upstream's "
        "per-model-family policy. Whatever it is, keep it identical across runs "
        "that will be compared.",
    )
    parser.add_argument(
        "--judges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the LLM judges in this pass instead of scoring separately.",
    )
    parser.add_argument("--stage-judge-model", default="anthropic/claude-haiku-4-5")
    parser.add_argument("--escalation-judge-model", default="anthropic/claude-sonnet-5")
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument(
        "--min-judge-entries",
        type=int,
        default=0,
        help="0 judges every run. Anything higher drops short runs, and whatever "
        "made them short decides which runs are dropped.",
    )
    parser.add_argument(
        "--retry-on-error",
        type=int,
        default=2,
        help="Retries for a sample that errors (transient sandbox/server faults).",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--display",
        default="plain",
        help="Inspect display. 'plain' is the one that behaves under nohup.",
    )
    args = parser.parse_args(argv)
    if (args.steer_vector is None) != (args.steer_strength is None):
        parser.error("--steer-vector and --steer-strength go together, or neither.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    task = ctfish(
        variant=args.variant,
        max_steps=args.max_steps,
        command_timeout=args.command_timeout,
        # Epochs are set HERE and nowhere else. `eval(epochs=...)` would win over
        # the task's value, so passing both is a way to run a different N than
        # the log's own task_args report.
        epochs=args.epochs,
        generate_config=args.generate_config,
        # None for a judge model drops that scorer.
        stage_judge_model=args.stage_judge_model if args.judges else None,
        escalation_judge_model=args.escalation_judge_model if args.judges else None,
        judge_samples=args.judge_samples,
        judge_temperature=args.judge_temperature,
        min_judge_entries=args.min_judge_entries,
    )

    model_args = {}
    metadata = {
        "variants": list(args.variant),
        "generate_config": args.generate_config,
        "judges_in_agent_pass": args.judges,
    }
    if args.steer_vector is not None:
        # In model_args because that is what the provider reads, and again in
        # metadata because a summary over many logs should not have to know
        # provider argument names to find out what was run.
        model_args = {
            "steer_vector": args.steer_vector,
            "steer_strength": args.steer_strength,
        }
        metadata.update(model_args)

    logs = eval(
        task,
        model=args.model,
        model_args=model_args,
        metadata=metadata,
        log_dir=args.log_dir,
        max_sandboxes=args.max_sandboxes,
        # A sample that crashes must not take the rest down with it: one lost
        # run is a missing data point, an aborted eval is a lost day.
        fail_on_error=False,
        retry_on_error=args.retry_on_error,
        log_level=args.log_level,
        display=args.display,
    )

    failed = False
    for log in logs:
        print(f"\n{log.eval.task}: {log.status}")
        if log.status == "error" and log.error:
            failed = True
            print(log.error.message)
        for score in log.results.scores if log.results else []:
            metrics = ", ".join(
                f"{name}={metric.value}" for name, metric in score.metrics.items()
            )
            print(f"  {score.name}: {metrics}")
        if log.results:
            # `unscored_samples` only exists on newer inspect_ai; reading it
            # unguarded turns a finished eval into a traceback and a nonzero exit.
            unscored = getattr(log.results, "unscored_samples", None)
            line = (
                f"  samples: completed={log.results.completed_samples}"
                f"/{log.results.total_samples}"
            )
            if unscored is not None:
                line += f" unscored={unscored}"
            print(line)
        print(f"  log: {log.location}")
    sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
