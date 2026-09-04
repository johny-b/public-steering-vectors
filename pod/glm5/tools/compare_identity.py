#!/usr/bin/env python3
"""Compare two probe runs token-for-token.  Gate 1 lives or dies here.

Checks, in increasing strength:
  * finish_reason and usage counts agree
  * the emitted text (reasoning_content + content) is byte-identical
  * the completion token id sequences are identical
  * the per-token logprobs are identical, and if not, by how much

A logprob difference of exactly zero on every token means the two servers ran
the same arithmetic, not merely that the argmax happened to agree -- which is
the claim gate 1 needs, because an argmax can agree while the forward pass has
drifted.
"""

from __future__ import annotations

import json
import sys


def dig(response: dict) -> dict:
    choice = response["choices"][0]
    message = choice.get("message", {}) or {}
    content = message.get("content") or ""
    # This wheel's reasoning parser puts the thinking in `reasoning`; older ones
    # use `reasoning_content`.  Read both so the comparison never silently drops
    # the part of the output that is most of it.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""

    token_ids = None
    for holder, key in ((choice, "token_ids"), (response, "token_ids"),
                        (message, "token_ids"), (choice, "completion_token_ids")):
        value = holder.get(key) if isinstance(holder, dict) else None
        if value:
            token_ids = list(value)
            break

    tokens, logprobs = [], []
    lp = choice.get("logprobs") or {}
    for entry in (lp.get("content") or []):
        tokens.append(entry.get("token"))
        logprobs.append(entry.get("logprob"))

    usage = response.get("usage", {}) or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "text": reasoning + "\x00" + content,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "finish_reason": choice.get("finish_reason"),
        "token_ids": token_ids,
        "tokens": tokens,
        "logprobs": logprobs,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": details.get("reasoning_tokens"),
    }


def main() -> None:
    a_path, b_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    by_id_b = {r["id"]: r for r in b["results"]}

    rows, all_ok = [], True
    for ra in a["results"]:
        rb = by_id_b.get(ra["id"])
        row: dict = {"id": ra["id"]}
        if not ra["ok"] or rb is None or not rb["ok"]:
            row.update(status="ERROR", detail=f"a_ok={ra['ok']} b_ok={rb and rb['ok']}")
            rows.append(row)
            all_ok = False
            continue

        da, db = dig(ra["response"]), dig(rb["response"])
        text_same = da["text"] == db["text"]
        ids_same = (
            (da["token_ids"] == db["token_ids"])
            if da["token_ids"] is not None
            else None
        )
        toks_same = da["tokens"] == db["tokens"]

        lp_a, lp_b = da["logprobs"], db["logprobs"]
        if lp_a and lp_b and len(lp_a) == len(lp_b):
            diffs = [abs(x - y) for x, y in zip(lp_a, lp_b)]
            max_lp = max(diffs) if diffs else 0.0
            n_lp_diff = sum(1 for d in diffs if d != 0.0)
        else:
            max_lp, n_lp_diff = None, None

        first_div = None
        if not text_same:
            for i, (x, y) in enumerate(zip(da["text"], db["text"])):
                if x != y:
                    first_div = i
                    break
            if first_div is None:
                first_div = min(len(da["text"]), len(db["text"]))

        ok = text_same and toks_same and (ids_same is not False) and (max_lp == 0.0)
        all_ok &= ok
        row.update(
            status="IDENTICAL" if ok else "DIFFERS",
            text_identical=text_same,
            token_ids_identical=ids_same,
            logprob_tokens_identical=toks_same,
            max_abs_logprob_delta=max_lp,
            n_tokens_with_logprob_delta=n_lp_diff,
            n_logprob_tokens=len(lp_a),
            finish_reason=[da["finish_reason"], db["finish_reason"]],
            prompt_tokens=[da["prompt_tokens"], db["prompt_tokens"]],
            completion_tokens=[da["completion_tokens"], db["completion_tokens"]],
            reasoning_tokens=[da["reasoning_tokens"], db["reasoning_tokens"]],
            chars=[da["reasoning_chars"] + da["content_chars"],
                   db["reasoning_chars"] + db["content_chars"]],
            first_divergent_char=first_div,
        )
        rows.append(row)

    summary = {
        "a": a.get("label"),
        "b": b.get("label"),
        "a_file": a_path,
        "b_file": b_path,
        "all_identical": bool(all_ok),
        "n": len(rows),
        "rows": rows,
    }
    json.dump(summary, open(out_path, "w"), indent=1)

    print(f"{a.get('label')}  vs  {b.get('label')}")
    print(
        f"{'id':18s} {'status':10s} {'tok':>6s} "
        f"{'cmpl_a/b':>13s} {'maxdlogp':>10s} {'ndiff':>6s}"
    )
    for r in rows:
        if r["status"] == "ERROR":
            print(f"{r['id']:18s} ERROR      {r['detail']}")
            continue
        ct = r["completion_tokens"]
        print(f"{r['id']:18s} {r['status']:10s} {r['n_logprob_tokens']:6d} "
              f"{str(ct[0]) + '/' + str(ct[1]):>13s} "
              f"{str(r['max_abs_logprob_delta']):>10s} "
              f"{str(r['n_tokens_with_logprob_delta']):>6s}")
    print(f"\nALL IDENTICAL: {all_ok}")


if __name__ == "__main__":
    main()
