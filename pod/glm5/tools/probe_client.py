#!/usr/bin/env python3
"""Run a prompt set against a local vLLM OpenAI server and dump everything.

Deliberately stdlib-only (urllib + threads) so it runs in the serving venv
without adding a dependency, and deliberately dumps the *raw* response object
rather than a summary: the identity gate needs token ids and logprobs, and a
summariser that drops them cannot be un-dropped after the server is gone.

`reasoning_effort` is never sent.  The vector was captured with the field
absent, which the GLM-5.3-Flash template renders as "Reasoning Effort: Max";
sending anything else would compare the steering against a different model state
from the one it was measured in.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one(base: str, model: str, item: dict, args) -> dict:
    messages = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    messages.append({"role": "user", "content": item["user"]})

    body = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": 1.0,
        "seed": args.seed,
        "max_tokens": int(item.get("max_tokens", args.max_tokens)),
        "stream": False,
    }
    if args.logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 0
    if args.token_ids:
        body["return_token_ids"] = True

    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = json.loads(response.read())
        return {
            "id": item["id"],
            "ok": True,
            "seconds": round(time.time() - started, 2),
            "response": raw,
        }
    except urllib.error.HTTPError as exc:
        return {
            "id": item["id"],
            "ok": False,
            "seconds": round(time.time() - started, 2),
            "error": f"HTTP {exc.code}",
            "body": exc.read().decode(errors="replace")[:4000],
        }
    except Exception as exc:  # noqa: BLE001 - the point is to record, not to classify
        return {
            "id": item["id"],
            "ok": False,
            "seconds": round(time.time() - started, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument("--token-ids", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    with open(args.prompts) as handle:
        items = json.load(handle)

    started = time.time()
    if args.concurrency <= 1:
        # Sequential on purpose for the identity gate: continuous batching makes
        # the reduction order depend on what else is in flight, so two servers
        # only compare cleanly if each request is alone in the engine.
        results = [one(args.base, args.model, item, args) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(
                pool.map(lambda it: one(args.base, args.model, it, args), items)
            )

    out = {
        "label": args.label,
        "prompts_file": args.prompts,
        "n": len(items),
        "n_ok": sum(1 for r in results if r["ok"]),
        "wall_seconds": round(time.time() - started, 1),
        "request_settings": {
            "temperature": args.temperature,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "logprobs": args.logprobs,
            "return_token_ids": args.token_ids,
            "reasoning_effort": "OMITTED (template default -> Max)",
        },
        "results": results,
    }
    with open(args.out, "w") as handle:
        json.dump(out, handle)
    print(
        f"[probe] {args.label}: {out['n_ok']}/{out['n']} ok in {out['wall_seconds']}s "
        f"-> {args.out}",
        flush=True,
    )
    for r in results:
        if not r["ok"]:
            print(
                f"[probe]   FAILED {r['id']}: "
                f"{r.get('error')} {r.get('body', '')[:300]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
