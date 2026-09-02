"""Model-holding capture worker.

Ported from local ``v2-steering-tools`` commits ea65b2f, a486b53, d732a8c,
6508dab, and 6678408. It performs exactly one eager offline prefill per prompt
and refuses any capture it cannot attribute to that prompt.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..capture_engine import config
from ..core import canonjson, digest, modelprofile

PROFILE = modelprofile.PROFILE
GPU_MEMORY_UTILIZATION = 0.85
CONTEXT_HEADROOM_TOKENS = 8
OK_LINE = "CAPTURE_WORKER_OK"
ASSERTIONS = (
    "capture-only environment asserted before model load; engine-init captures "
    "purged and counted; exactly one new capture per prompt; sidecar num_tokens "
    "equals prompt token count; every prompt checked and at least one captured"
)


class WorkerError(RuntimeError):
    """A worker refusal raised before a misleading result can be written."""


@dataclass(frozen=True)
class WorkerSpec:
    prompts: list[Any]
    capture_dir: str
    result_path: str
    model: str = PROFILE.model_id
    max_model_len: int = 4096
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompts": list(self.prompts),
            "capture_dir": self.capture_dir,
            "result_path": self.result_path,
            "model": self.model,
            "max_model_len": int(self.max_model_len),
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
        }

    def write(self, path: str | os.PathLike[str]) -> Path:
        return canonjson.write_json(path, self.as_dict())

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> "WorkerSpec":
        record = canonjson.read_json(path)
        if not isinstance(record, dict):
            raise WorkerError("worker spec must be a JSON object")
        allowed = {
            "prompts",
            "capture_dir",
            "result_path",
            "model",
            "max_model_len",
            "gpu_memory_utilization",
        }
        unknown = set(record) - allowed
        if unknown:
            raise WorkerError(f"unknown spec fields {sorted(unknown)}")
        return cls(**record)


def check_environment(
    env: Mapping[str, str], capture_dir: str | os.PathLike[str]
) -> config.CaptureConfig:
    cfg = config.read_config(env)
    if cfg is None:
        raise WorkerError("this process is not configured to capture")
    if Path(cfg.capture_dir).resolve() != Path(capture_dir).resolve():
        raise WorkerError("engine and worker capture directories differ")
    return cfg


def context_length(token_counts: Sequence[int], floor: int) -> int:
    if not token_counts:
        raise WorkerError("no prompts: nothing to size a context for")
    return max(int(floor), max(int(n) for n in token_counts) + CONTEXT_HEADROOM_TOKENS)


def new_capture_index(
    before: Sequence[int], after: Sequence[int], *, prompt: int
) -> int:
    fresh = sorted(set(int(value) for value in after) - set(int(value) for value in before))
    if len(fresh) == 1:
        return fresh[0]
    if not fresh:
        raise WorkerError(f"prompt {prompt} published no capture")
    raise WorkerError(f"prompt {prompt} published {len(fresh)} captures, not one")


def check_capture(
    record: Mapping[str, Any], token_ids: Sequence[int], *, prompt: int
) -> None:
    config.validate_sidecar(record, where=f"capture for prompt {prompt}")
    if int(record["num_tokens"]) != len(token_ids):
        raise WorkerError(
            f"prompt {prompt}: capture has {record['num_tokens']} rows but prompt "
            f"has {len(token_ids)} tokens; last row is not the last prompt token"
        )


def check_ran(
    n_checked: int, n_prompts: int, *, published: Sequence[int]
) -> None:
    if n_prompts < 1:
        raise WorkerError("no prompts, so no forward pass ran")
    if n_checked != n_prompts:
        raise WorkerError(f"{n_checked} of {n_prompts} prompts were checked")
    if len(published) < n_prompts:
        raise WorkerError("published files do not evidence every prompt")


def templated(tokenizer: Any, prompt: Any) -> str:
    messages = (
        [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else list(prompt)
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def tokenize_prompts(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    explicit = [
        list(tokenizer(text, add_special_tokens=False).input_ids) for text in texts
    ]
    default = [list(tokenizer(text).input_ids) for text in texts]
    for index, (without, with_default) in enumerate(zip(explicit, default)):
        if len(without) != len(with_default):
            raise WorkerError(
                f"prompt {index}: tokenizer adds special tokens to templated text"
            )
    return explicit


def run(
    spec: WorkerSpec, *, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    check_environment(environment, spec.capture_dir)
    if spec.model != PROFILE.model_id:
        raise WorkerError(
            f"capture worker is profiled only for {PROFILE.model_id!r}, "
            f"got {spec.model!r}"
        )
    if not spec.prompts:
        check_ran(0, 0, published=[])

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(spec.model)
    texts = [templated(tokenizer, prompt) for prompt in spec.prompts]
    id_lists = tokenize_prompts(tokenizer, texts)
    token_counts = [len(ids) for ids in id_lists]
    max_model_len = context_length(token_counts, spec.max_model_len)
    llm = LLM(
        model=spec.model,
        dtype=PROFILE.dtype,
        max_model_len=max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enforce_eager=True,
        gpu_memory_utilization=spec.gpu_memory_utilization,
    )
    purged = config.purge_captures(spec.capture_dir)
    params = SamplingParams(temperature=0.0, max_tokens=1)
    records: list[dict[str, Any]] = []
    for index, ids in enumerate(id_lists):
        before = config.capture_indices(spec.capture_dir)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], params)
        after = config.capture_indices(spec.capture_dir)
        capture_index = new_capture_index(before, after, prompt=index)
        sidecar = config.read_sidecar(
            config.capture_sidecar_path(spec.capture_dir, capture_index)
        )
        check_capture(sidecar, ids, prompt=index)
        records.append(
            {
                "prompt": index,
                "capture_index": capture_index,
                "n_prompt_tokens": len(ids),
                "capture_num_tokens": int(sidecar["num_tokens"]),
                "pid": int(sidecar["pid"]),
                "templated_sha256": digest.sha256_text(texts[index]),
            }
        )
    published = config.capture_indices(spec.capture_dir)
    check_ran(len(records), len(spec.prompts), published=published)
    result = {
        "model": spec.model,
        "capture_dir": spec.capture_dir,
        "n_prompts": len(spec.prompts),
        "n_checked": len(records),
        "capture_files_from_engine_init": len(purged),
        "max_model_len": max_model_len,
        "prompt_token_lengths": token_counts,
        "templated_first_prompt": texts[0],
        "capture_indices": published,
        "records": records,
        "pid": os.getpid(),
        "assertions": ASSERTIONS,
    }
    canonjson.write_json(spec.result_path, result)
    print(OK_LINE, flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m steering_vectors.build.worker <spec.json>",
            file=sys.stderr,
        )
        return 2
    try:
        run(WorkerSpec.read(args[0]))
    except (WorkerError, config.CaptureConfigError) as exc:
        print(f"CAPTURE_WORKER_REFUSED {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
