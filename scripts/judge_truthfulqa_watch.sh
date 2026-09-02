#!/usr/bin/env bash
#
# Judge the TruthfulQA sweep as it finishes, rather than once at the end.
#
#     setsid nohup scripts/judge_truthfulqa_watch.sh \
#         > /work/workspace/judge-logs/judge.log 2>&1 < /dev/null &
#
# The sweep writes one .eval per strength and is still running, so a single
# judging pass sees only the strengths that happened to be finished when it
# started. This runs `judge_truthfulqa.py` over and over instead. Each pass
# skips logs that are not `success` and skips sample ids already in their
# condition's JSONL file, so a pass costs nothing where there is nothing new
# and picks up a strength within a couple of minutes of it landing.
#
# It stops on its own, two ways, and says which in the marker file it leaves
# behind: all nine conditions complete, or nothing new for long enough that the
# sweep is clearly not producing any more. A watcher that only knew how to
# finish would spin until the session died if the sweep wedged.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

OUT_DIR="logs/truthfulqa-judge"
MARKER="$OUT_DIR/DONE"
CONDITIONS=9        # vectors x strengths in the sweep
ROWS_PER_CONDITION=817
SLEEP_SECONDS=120
MAX_IDLE_PASSES=30  # ~1h of passes that judged nothing

mkdir -p "$OUT_DIR"

# A marker left by an earlier watcher would otherwise be read as this one's
# result the moment this one starts.
if [[ -e "$MARKER" ]]; then
    echo "removing stale marker $MARKER"
    rm -f "$MARKER"
fi

# Rows per condition file, as `<stem> <count>` lines.
counts() {
    local file
    for file in "$OUT_DIR"/*.jsonl; do
        [[ -e "$file" ]] || continue
        printf '%s %s\n' "$(basename "$file" .jsonl)" "$(wc -l < "$file" | tr -d ' ')"
    done
}

total_rows() {
    counts | awk '{sum += $2} END {print sum + 0}'
}

# Complete means nine files that each hold a whole condition. `>=` rather than
# `==` so that a condition with more samples than expected finishes rather than
# spinning to the idle limit.
complete() {
    local done_files
    done_files=$(counts | awk -v n="$ROWS_PER_CONDITION" '$2 >= n' | wc -l)
    [[ "$done_files" -ge "$CONDITIONS" ]]
}

write_marker() {
    {
        echo "$1"
        echo "finished: $(date -Is)"
        echo "passes: $pass"
        echo
        echo "condition rows"
        counts
        echo
        echo "total $(total_rows)"
    } > "$MARKER"
    echo "--- wrote $REPO/$MARKER ---"
    cat "$MARKER"
}

pass=0
idle=0
while true; do
    pass=$((pass + 1))
    before=$(total_rows)
    echo "=== pass $pass ($(date -Is)), $before rows so far ==="

    python scripts/judge_truthfulqa.py || echo "  pass $pass exited $?; continuing"

    after=$(total_rows)
    if [[ "$after" -gt "$before" ]]; then
        idle=0
    else
        idle=$((idle + 1))
        echo "  no new judgements ($idle/$MAX_IDLE_PASSES idle passes)"
    fi

    if complete; then
        write_marker "complete: all $CONDITIONS conditions judged"
        exit 0
    fi
    if [[ "$idle" -ge "$MAX_IDLE_PASSES" ]]; then
        write_marker "gave up: $MAX_IDLE_PASSES consecutive passes judged nothing new"
        exit 1
    fi

    sleep "$SLEEP_SECONDS"
done
