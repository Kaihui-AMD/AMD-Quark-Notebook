"""Small vLLM smoke test helper for exported Quark models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Local Hugging Face or exported Quark model directory.")
    parser.add_argument("--quantization", default="", help="Optional vLLM quantization hint, for example quark.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to generate from. Can be repeated.")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--dtype", default="", help="Optional vLLM dtype override, for example float16.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable graph/compile paths for smoke tests.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default="", help="Optional JSON path for prompts and outputs.")
    return parser.parse_args()


def build_prompts(model_dir: str, prompts: list[str], use_chat_template: bool, trust_remote_code: bool) -> list[str]:
    if prompts:
        raw_prompts = prompts
    else:
        raw_prompts = [
            "请用一句话解释 SmoothQuant。",
            "What is model quantization?",
        ]

    if not use_chat_template:
        return raw_prompts

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    prompts = build_prompts(
        model_dir=args.model_dir,
        prompts=args.prompt,
        use_chat_template=args.use_chat_template,
        trust_remote_code=args.trust_remote_code,
    )

    kwargs: dict[str, object] = {
        "model": args.model_dir,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.quantization:
        kwargs["quantization"] = args.quantization
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.enforce_eager:
        kwargs["enforce_eager"] = True

    print("Loading model:", args.model_dir)
    print("vLLM kwargs:", {key: value for key, value in kwargs.items() if key != "model"})
    llm = LLM(**kwargs)
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
    )

    records = []
    for output in outputs:
        generated = output.outputs[0]
        record = {
            "prompt": output.prompt,
            "text": generated.text,
            "finish_reason": generated.finish_reason,
            "num_output_tokens": len(generated.token_ids),
        }
        records.append(record)
        print("=" * 80)
        print("PROMPT:", output.prompt)
        print("OUTPUT:", repr(generated.text))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        print("Saved outputs to:", output_path)


if __name__ == "__main__":
    main()
