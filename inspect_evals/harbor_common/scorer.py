"""The one seam between this repository and `inspect_harbor`'s scorer.

Every harbor-backed eval here (Terminal-Bench 2.1, and SWE-bench Pro after it)
scores through `harbor_reward`, which wraps the adapter's own scorer, and
reports `harbor_diagnostics` beside it. There is one copy of this module on
purpose: it is a wrapper around third-party code that was audited line by
line, and a second copy would be a second thing to re-audit every time the
adapter is bumped.

Five findings from that audit shape everything below. All five were confirmed
against inspect-harbor 0.7.4 and inspect_ai 0.3.259, at the lines cited.

1. The adapter's scorer raises on verifier infrastructure failure rather than
   returning a low score. It defines four exception classes of its own
   (`CopyTestsDirError`, `VerifierOutputParseError`, `RewardFileNotFoundError`,
   `RewardFileEmptyError`) at `inspect_harbor/_harbor/scorer.py:25-37` and
   raises them uncaught through `:63-181`, and the sandbox `exec` that runs
   the verifier (`:100-105`) raises `TimeoutError` and `PermissionError`
   straight through as well. Inspect runs scorers in a plain loop with no
   per-scorer try (`inspect_ai/_eval/task/run.py:2167-2186`), so any of those
   aborts the loop: the sample is errored and every scorer after the raising
   one never runs. A container that could not be reached and an agent that
   failed the task would then be the same event in the results table, and the
   diagnostics that would tell them apart would be missing exactly when they
   are needed.

   So `harbor_reward` catches `Exception`, reports NaN per key, and records
   what happened in the store. `verifier_failed` on the diagnostics table is
   the number to read first in any harbor run.

2. The adapter deletes `/tests` and `/logs/verifier` from the sandbox
   immediately after building its `Score`
   (`inspect_harbor/_harbor/scorer.py:117-118`, implemented at
   `_harbor/sandbox_utils.py:87-91`), so `reward.txt` and the structured CTRF
   results are gone by the time anything downstream could look. The one
   surviving record of what the verifier did is the `Score.explanation` the
   adapter builds at `_harbor/scorer.py:113`, which carries the exit code plus
   the verifier's full stdout and stderr. `harbor_reward` lifts it into
   `Score.metadata` verbatim and never truncates it.

3. The adapter's headline convention is `passed = reward > 0`
   (`inspect_harbor/_harbor/scorer.py:108`, used at `:112`). This module does
   not inherit it. `resolved` here is `reward == 1.0`, because on these
   benchmarks a task is solved or it is not, and "some tests passed" is not a
   solved task. On Terminal-Bench 2.1 the two conventions agree by
   construction (all 89 verifiers write a bare 0 or 1), which is why
   `reward_fractional` is a diagnostic and not a scoring rule: it is the
   canary that says a dataset has started reporting partial credit and that
   the choice between the two conventions has become load-bearing.

4. Inspect's own `react` agent, which is the scaffold the adapter ships
   (`inspect_harbor/_harbor/task.py:97-104`), deletes the submit tool call
   from `state.messages` before scoring: `_remove_submit_tool` at
   `inspect_ai/agent/_react.py:383-388`, because `AgentSubmit.keep_in_messages`
   defaults to False. A scorer that reads the messages therefore cannot see
   that the agent submitted at all. The call survives in the transcript as a
   `ToolEvent`, so the `submitted` and `tool_calls` diagnostics read the
   transcript first and fall back to the messages; see `_tool_calls`.

5. The adapter's oracle solver throws away the reference solution's result.
   It awaits `sandbox().exec` on the task's own solve script and discards the
   `ExecResult` -- exit code, stdout and stderr
   (`inspect_harbor/_harbor/solver.py:53-60`). On an oracle sweep the reward
   is then the only signal, and a reference solution that died half way
   through looks exactly like one the verifier ran and failed. Inspect records
   the exit code as `SandboxEvent.result` even though the adapter ignores it
   (`inspect_ai/event/_sandbox.py:31-32`), so `solution_exit_code` reads it
   back off the transcript; see `_solution_exit_code`. The output cannot be
   recovered the same way -- inspect keeps only its first 20 lines
   (`inspect_ai/util/_sandbox/events.py:239-243`).

Unscorable samples get NaN per key rather than a whole-sample
`Score.unscored()`. Inspect counts a per-key NaN toward that key's
`unscored_samples` and keeps the key set constant, so a broken run and a clean
one have the same table shape and the difference between them is a number.

That last sentence holds at `epochs=1`. It does not survive epoch reduction:
inspect's reducers drop non-reducible values before applying the statistic
(`inspect_ai/scorer/_reducer/reducer.py:456-460`), so a task whose verifier
failed on one epoch of two reduces to the other epoch's reward and is counted
as fully scored. At `epochs > 1` the infrastructure-failure rate is the
`verifier_failed` mean on the diagnostics table, not `reward.unscored_samples`.
"""

from __future__ import annotations

import math
import re
from typing import Any

from inspect_ai.event import (
    Event,
    ModelEvent,
    SampleLimitEvent,
    SandboxEvent,
    ToolEvent,
)
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageAssistant
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import StoreModel
from pydantic import Field

NAN = float("nan")

REWARD_KEYS = ("reward", "resolved")
"""Every key `harbor_reward` reports, so a sample whose verifier never ran can
report all of them as NaN rather than reporting none of them."""

TRUNCATION_STOP_REASONS = ("max_tokens", "model_length")
"""Stop reasons that mean a generation stopped before the model had finished.

Both, not just the first. "max_tokens" is the budget the run asked for and
"model_length" is the served context window; a reply cut off by either ended
mid-thought, but the fixes differ (raise the budget, or serve more context),
so the diagnostics split them into their own keys as well as counting them
together.
"""

DEFAULT_SUBMIT_TOOL = "submit"
"""The name inspect's `react` agent gives its submission tool.

The `submitted` diagnostic looks for a call by this name, in the transcript
first and in the messages second; see finding 4 in the module docstring for why
the messages alone are not enough. A `--solver` that ends its turn some other
way -- a different submit tool name, or a scaffold that just stops -- reads as
`submitted = 0`, which is a fact about the scaffold rather than about the
model. Read it next to `messages` and `tool_calls`.
"""

EXIT_CODE_PATTERN = re.compile(r"^Test exit code: (-?\d+)")
"""The first line of the adapter's explanation string.

Parsed rather than trusted: the exit code is a useful thing to have on a
failed verifier, and the explanation is the only place the adapter puts it.
If the adapter's format changes the parse simply misses and the exit code is
reported as absent, which is why nothing scored depends on it.
"""

SOLUTION_DIR = "/solution/"
"""Where `inspect_harbor`'s oracle solver puts the task's reference solution.

The solver copies the whole solution directory to `/solution` and runs the
task's own script from inside it (`inspect_harbor/_harbor/solver.py:47,53-60`),
so an exec whose command names a path under this prefix is the reference
solution running and nothing else is. Used by `_solution_exit_code`.
"""


class HarborState(StoreModel):
    """What `harbor_reward` leaves behind for `harbor_diagnostics`.

    The two scorers are separate so that a verifier failure cannot take the
    diagnostics down with it, and the store is how the second one learns what
    happened in the first. Nothing here is derived at read time: a scorer that
    recomputed a fact could disagree with the score that was actually
    reported.
    """

    scored: bool = Field(default=False)
    """Whether `harbor_reward` ran at all.

    False in the diagnostics means the scorer stack did not run in the
    expected order (or at all), which is itself worth seeing rather than
    silently reading as "no failures".
    """
    reward: float | None = Field(default=None)
    """The adapter's raw reward, or None when the verifier did not produce one."""
    reward_dict: dict[str, Any] | None = Field(default=None)
    """The parsed `reward.json` body, when the verifier wrote JSON.

    None on the `reward.txt` path, which is the only path Terminal-Bench 2.1
    takes. Kept because the adapter's JSON branch silently uses the first
    value in the object when there is no "reward" key, and this is the record
    of which values it was choosing between.
    """
    verifier_failed: bool = Field(default=False)
    """Whether the adapter's scorer raised instead of returning a reward."""
    verifier_error_type: str = Field(default="")
    """The exception class name, e.g. "TimeoutError" or "RewardFileEmptyError"."""
    verifier_error: str = Field(default="")
    """The exception text, complete."""
    verifier_exit_code: int | None = Field(default=None)
    """The verifier script's exit code, when the explanation carried one."""
    verifier_output_chars: int = Field(default=0)
    """Size of the verifier's stdout+stderr record, in characters."""
    adapter_answer: str = Field(default="")
    """The adapter's own verdict string, "PASS" or "FAIL".

    Recorded, not used. It is the `reward > 0` convention this module declines
    to inherit, and keeping it beside `resolved` makes any future divergence
    between the two visible in the log rather than a matter of argument.
    """


def _exit_code(explanation: str | None) -> int | None:
    """The verifier's exit code, read out of the adapter's explanation string."""
    if not explanation:
        return None
    match = EXIT_CODE_PATTERN.match(explanation)
    return int(match.group(1)) if match else None


def _events() -> list[Event] | None:
    """This sample's complete transcript events, or None if they cannot be read.

    Two ways they cannot be read, and both have to produce None rather than a
    partial list, because a partial list would be reported as a count and a
    count is indistinguishable from a small one.

    Inspect can run a transcript in bounded-memory mode, where older events are
    evicted. If a history provider is attached they are served lazily from it,
    and reading can raise if it has already been torn down. If no provider is
    attached, `Transcript.events` returns the resident tail *without raising*
    (`inspect_ai/log/_transcript.py:498-502`, truncation flagged at `:748`) --
    a silent undercount on exactly the long agentic samples bounded mode
    exists for. `TranscriptHistory.full_history_available`
    (`_transcript.py:233-243`) is the property that tells the two apart, so it
    is checked before the events are read.

    Scoring must not be the thing that errors a sample, so anything unexpected
    here is also None and the caller falls back to what `TaskState` carries.
    """
    try:
        current = transcript()
        if not current.history.full_history_available:
            return None
        return list(current.events)
    except Exception:  # noqa: BLE001 - a diagnostics read must never error a sample
        return None


def _generation_truncations(
    state: TaskState, events: list[Event] | None
) -> dict[str, float]:
    """Count generations that stopped at a limit rather than at an answer.

    An agentic sample is many generations, so these are counts and not flags:
    the mean of `truncated_generations` is truncations per sample, not the
    share of samples with one. When the transcript is unavailable the counts
    fall back to the final generation alone, which is a floor and not a
    guess -- `truncation_source` in the metadata says which reading was used.
    """
    if events is not None:
        outputs = [
            event.output.stop_reason
            for event in events
            if isinstance(event, ModelEvent) and event.output is not None
        ]
    else:
        outputs = [state.output.stop_reason] if state.output is not None else []
    return {
        "truncated_generations": float(
            sum(1 for reason in outputs if reason in TRUNCATION_STOP_REASONS)
        ),
        "max_tokens_stops": float(sum(1 for r in outputs if r == "max_tokens")),
        "model_length_stops": float(sum(1 for r in outputs if r == "model_length")),
        "generations": float(len(outputs)),
    }


def _tool_calls(
    state: TaskState, events: list[Event] | None
) -> tuple[float, float, str]:
    """Tool calls, whether the agent submitted, and which record said so.

    The messages alone cannot answer either question under the adapter's own
    scaffold: `react` strips the submit tool call out of `state.messages`
    before scoring (finding 4 in the module docstring), so a message scan
    reports `submitted = 0` on a sample where the agent did submit, and
    undercounts `tool_calls` by one for the same reason. The transcript keeps
    the call as a `ToolEvent`, so it is the primary record here.

    The message scan is still consulted, and a submit found in either record
    counts. A scaffold that keeps its submit call in the messages
    (`AgentSubmit(keep_in_messages=True)`, or a hand-written agent) is then
    read correctly even when the transcript is unavailable. When neither record
    can answer -- no transcript, and nothing in the messages -- `submitted` is
    NaN rather than 0, because "did not submit" and "could not tell" are
    different claims. `tool_calls` falls back to the message count, which is a
    floor rather than a guess, and the returned source string says which
    reading was used.
    """
    from_messages = [
        call
        for message in state.messages
        if isinstance(message, ChatMessageAssistant)
        for call in (message.tool_calls or [])
    ]
    submitted_in_messages = any(
        call.function == DEFAULT_SUBMIT_TOOL for call in from_messages
    )
    if events is None:
        return (
            float(len(from_messages)),
            1.0 if submitted_in_messages else NAN,
            "messages only",
        )
    functions = [event.function for event in events if isinstance(event, ToolEvent)]
    return (
        float(len(functions)),
        float(submitted_in_messages or DEFAULT_SUBMIT_TOOL in functions),
        "transcript",
    )


def _limit_types(events: list[Event] | None) -> list[str] | None:
    """The types of any sample limits that fired, or None if unreadable."""
    if events is None:
        return None
    return [event.type for event in events if isinstance(event, SampleLimitEvent)]


def _solution_exit_code(events: list[Event] | None) -> float:
    """The exit code of the reference solution, on a run that used the oracle.

    The adapter's oracle solver awaits `sandbox().exec` on the task's own
    solve script and then throws the whole result away -- exit code, stdout and
    stderr alike (`inspect_harbor/_harbor/solver.py:53-60`). Nothing downstream
    is told whether the reference solution finished, so a solution that died
    part-way and a solution that ran to completion and was failed by the
    verifier arrive as the same `reward = 0`. Those are opposite findings: the
    first is a broken environment, the second is a broken task.

    Inspect keeps the exit code even though the adapter does not: every sandbox
    call is recorded as a `SandboxEvent`, and `result` is the exit status
    (`inspect_ai/event/_sandbox.py:31-32`). So this reads it back out of the
    transcript, matching on the `/solution/` prefix the solver copies to. That
    is the whole fix -- the solver is left exactly as audited, and the fact it
    discards is recovered rather than recomputed.

    The event's `output` is not recovered with it, because inspect keeps only
    the first 20 lines and 10,000 characters of it
    (`inspect_ai/util/_sandbox/events.py:239-243`, via
    `inspect_ai/_util/text.py:288-298`). On a solution that installs packages
    for five minutes before failing, the error is past both caps and no
    downstream consumer can get it back. The exit code is what survives, so the
    exit code is what this reports.

    NaN, not 0, when there is no such exec: an agent run has no reference
    solution, and an unreadable transcript cannot say. Both are "no oracle
    ran here", which is a different claim from "the oracle succeeded". The
    last matching exec wins, so a solver that shells into `/solution` more than
    once reports the run that mattered.
    """
    if events is None:
        return NAN
    codes = [
        event.result
        for event in events
        if isinstance(event, SandboxEvent)
        and event.action == "exec"
        and SOLUTION_DIR in (event.cmd or "")
        and event.result is not None
    ]
    return float(codes[-1]) if codes else NAN


def _declared_agent_timeout(state: TaskState) -> float:
    """The task's own `[agent].timeout_sec`, which inspect_harbor does not enforce.

    harbor's own runner enforces this and raises `AgentTimeoutError` when it
    expires. inspect_harbor reads the verifier timeout out of the same config
    and ignores the agent one: it sets no `time_limit` on the Task at all. So
    an agentic run through this adapter is unbounded where the official
    harness is bounded, which makes it a materially easier benchmark unless
    the operator sets a limit. Reported per sample so the gap is visible in
    the results table rather than only in a README. NaN when the task declares
    no timeout, since 0 would read as "no time allowed".
    """
    config = state.metadata.get("harbor_config")
    if not isinstance(config, dict):
        return NAN
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return NAN
    timeout = agent.get("timeout_sec")
    return float(timeout) if isinstance(timeout, (int, float)) else NAN


@scorer(metrics={"reward": [mean(), stderr()], "resolved": [mean(), stderr()]})
def harbor_reward(inner: Scorer, verifier_login_shell: bool = True) -> Scorer:
    """Run the adapter's scorer, and survive it failing.

    Args:
        inner: The adapter's scorer, normally `inspect_harbor.harbor_scorer()`.
            Passed in rather than imported so that this module stays
            importable on an interpreter with no `inspect_harbor` installed --
            the guard module's job is to produce the actionable error about
            that, and it cannot do it from behind an import that already
            failed.
        verifier_login_shell: Whether to leave the adapter's login-shell
            verifier invocation alone. `True` is the adapter's own behaviour
            and the default, because changing how a benchmark's verifier is
            invoked is a deviation and deviations are opt-in here.

            `False` makes the verifier run under the image's own `PATH`
            instead, and SWE-bench Pro passes it because it has to: the
            adapter runs `["bash", "-l", <test.sh>]`
            (`inspect_harbor/_harbor/scorer.py:100`), `/etc/profile` in the
            `jefzda/sweap-images` images overwrites the `PATH` the image sets
            through `ENV`, and `/usr/local/go/bin` is one of the entries it
            drops. Every Go task -- 280 of the 731 -- then scores a clean 0
            with `verifier_failed = 0`, which is indistinguishable from a
            capability result. See `harbor_common.verifier_shell` for how that
            was established, and for the evidence that Terminal-Bench 2.1 is
            not affected and therefore keeps `True`.

            The override is installed here, at scorer construction, and is
            never removed: inspect scores samples concurrently, so unpatching
            per call would pull the override out from under a sample that was
            still inside the adapter's scorer.

    Two keys.

    `reward` is the adapter's raw float, unrounded and unclamped. On
    Terminal-Bench 2.1 it is 0.0 or 1.0 for every task, but the adapter can
    return anything the verifier writes, so it is reported as it came.

    `resolved` is `reward == 1.0`. That is deliberately not the adapter's own
    `reward > 0` rule; see the module docstring. Where a dataset only ever
    writes 0 or 1 the two agree, and `reward_fractional` in the diagnostics is
    the alarm for the day that stops being true.

    Both are NaN when the verifier could not be run or its output could not be
    read, and neither is ever zero for that reason. Scoring an unreachable
    container as a failed task would let an infrastructure problem look like a
    capability result -- and under a steering sweep, where container behaviour
    can shift with the workload, it would look like a dose-response curve.

    The adapter's explanation (exit code, plus the verifier's complete stdout
    and stderr) is lifted into `metadata["verifier_output"]` before it is
    returned, because the adapter has already deleted `/logs/verifier` and
    `/tests` from the sandbox by the time this returns. It is never truncated.

    Raises:
        ImportError: If `verifier_login_shell` is False and `inspect_harbor`
            is not installed.
        RuntimeError: If `verifier_login_shell` is False and the installed
            adapter no longer runs the verifier in a login shell, in which case
            the override would be a silent no-op and the Go tasks would go back
            to scoring a confident zero.
    """
    if not verifier_login_shell:
        # Imported here rather than at module scope: this package is required
        # to stay importable on an interpreter with no `inspect_harbor`, and
        # installing the override means reaching into the adapter.
        from .verifier_shell import install_image_path_verifier_shell

        install_image_path_verifier_shell()

    async def score(state: TaskState, target: Target) -> Score:
        store = state.store_as(HarborState)
        store.scored = True
        try:
            result = await inner(state, target)
        # Broad on purpose. The adapter raises four exception classes of its
        # own, and the sandbox exec beneath it raises TimeoutError,
        # PermissionError and OSError straight through -- seven types, of
        # which the likeliest here is the verifier timing out. All of them
        # mean the same thing to this eval: unscored, not failed. See finding
        # 1 in the module docstring.
        except Exception as error:  # noqa: BLE001 - see module docstring
            store.verifier_failed = True
            store.verifier_error_type = type(error).__name__
            store.verifier_error = str(error)
            return Score(
                value={key: NAN for key in REWARD_KEYS},
                explanation=(
                    f"Verifier infrastructure failure "
                    f"({type(error).__name__}): {error}"
                ),
                metadata={
                    "verifier_failed": True,
                    "verifier_error_type": type(error).__name__,
                    "verifier_error": str(error),
                },
            )

        reward = float(result.value) if isinstance(result.value, (int, float)) else NAN
        explanation = result.explanation or ""
        inner_metadata = result.metadata or {}
        reward_dict = inner_metadata.get("reward_dict")

        store.reward = None if math.isnan(reward) else reward
        store.reward_dict = reward_dict if isinstance(reward_dict, dict) else None
        store.verifier_exit_code = _exit_code(explanation)
        store.verifier_output_chars = len(explanation)
        store.adapter_answer = result.answer or ""
        # A reward the adapter returned in a shape that is not a number is not
        # a zero: it is the same "cannot be read" outcome as a raise, and it
        # is recorded as one so it lands in `verifier_failed` too.
        if math.isnan(reward):
            store.verifier_failed = True
            store.verifier_error_type = "NonNumericReward"
            store.verifier_error = f"Adapter returned Score.value={result.value!r}"

        return Score(
            value={
                "reward": reward,
                "resolved": NAN if math.isnan(reward) else float(reward == 1.0),
            },
            answer=result.answer,
            explanation=explanation,
            # The adapter's own metadata is spread first rather than read for
            # the one key we know about. Against 0.7.4 it sets `reward_dict`
            # and nothing else (`_harbor/scorer.py:114`), so this is currently
            # a no-op -- but this module is re-audited on every adapter bump,
            # and a key a later adapter adds should show up in the log to be
            # noticed rather than disappear here in silence.
            metadata={
                **inner_metadata,
                "verifier_output": explanation,
                "verifier_exit_code": store.verifier_exit_code,
                "adapter_answer": result.answer,
                "reward_dict": store.reward_dict,
            },
        )

    return score


@scorer(metrics={"*": [mean(), stderr()]})
def harbor_diagnostics() -> Scorer:
    """What happened to the sample, whatever the reward came out as.

    Every key is always present, so two runs always have the same table and
    the difference between them is arithmetic rather than interpretation.

    `verifier_failed` is the first thing to read in any harbor run: it is the
    share of samples where the container, the test copy or the reward file
    failed, and those samples contribute nothing to `reward` or `resolved`. A
    non-zero value is not a model result. `unscored` is the same fact under
    the name the other evals in this repository use for it, so a results table
    can carry one column across evals that do not otherwise share code.

    `reward_fractional` is the canary described in the module docstring: 1 when
    the verifier reported partial credit, which no Terminal-Bench 2.1 task
    does. It is NaN, not 0, when there is no reward to classify.

    `messages`, `tool_calls` and `submitted` are the shape of the agent's run.
    They are how a steering condition that made the model quit early is told
    apart from one that made it fail the task: the same `resolved = 0` with a
    tenth of the tool calls is a different finding. `tool_calls` and
    `submitted` read the transcript, not the message log, because `react`
    deletes its own submit call from the messages before scoring; see
    `_tool_calls`. `submitted` is NaN when neither record can answer.

    `agent_limit_hit` counts sample limits that fired (message, token, time,
    working, turn, cost or an operator interrupt). It reads the transcript,
    and reports NaN rather than 0 when the transcript cannot be read, because
    "no limit fired" and "could not tell" are different claims.

    `declared_agent_timeout_sec` is the task's own `[agent].timeout_sec`,
    which this adapter does not enforce; see `_declared_agent_timeout`.

    `solution_exit_code` is how an oracle run says whether the reference
    solution itself survived. The adapter's oracle solver discards it, so a
    reference solution that died half way through and one the verifier simply
    failed both arrive as `reward = 0`; this reads the code back out of the
    transcript and tells them apart. It is NaN on an agent run, which has no
    reference solution to run. See `_solution_exit_code`.
    """

    async def score(state: TaskState, target: Target) -> Score:
        store = state.store_as(HarborState)
        reward = store.reward
        events = _events()
        limits = _limit_types(events)
        truncations = _generation_truncations(state, events)
        tool_calls, submitted, tool_call_source = _tool_calls(state, events)
        return Score(
            value={
                "verifier_failed": float(store.verifier_failed),
                "unscored": float(store.verifier_failed),
                "scorer_ran": float(store.scored),
                "reward_fractional": (
                    NAN if reward is None else float(reward not in (0.0, 1.0))
                ),
                "messages": float(len(state.messages)),
                "tool_calls": tool_calls,
                "submitted": submitted,
                "agent_limit_hit": NAN if limits is None else float(bool(limits)),
                **truncations,
                "solution_exit_code": _solution_exit_code(events),
                "declared_agent_timeout_sec": _declared_agent_timeout(state),
                "verifier_output_chars": float(store.verifier_output_chars),
            },
            metadata={
                "verifier_error_type": store.verifier_error_type,
                "verifier_error": store.verifier_error,
                "verifier_exit_code": store.verifier_exit_code,
                "adapter_answer": store.adapter_answer,
                "limit_types": limits,
                "truncation_source": (
                    "transcript" if events is not None else "final generation only"
                ),
                "tool_call_source": tool_call_source,
                "task_name": state.metadata.get("task_name"),
            },
        )

    return score
