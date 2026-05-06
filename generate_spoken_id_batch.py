#!/usr/bin/env python3
"""
Generate repeated audio files for alphanumeric IDs using the local OpenAI-compatible TTS API.

Examples:
    python generate_spoken_id_batch.py --id S199279817
    python generate_spoken_id_batch.py --id S199279817 --count 10 --output-dir output/ids
    python generate_spoken_id_batch.py --input-file synthesis_txt.txt --count 10

For example:
    S199279817 -> S一九九二七九八一七
    Output files: S199279817_01.wav, S199279817_02.wav, ...
"""

import argparse
import datetime
import json
import time
from pathlib import Path
from urllib import error, request


CHINESE_DIGITS = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repeated TTS audio for alphanumeric IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--id",
        dest="single_id",
        help="Single alphanumeric ID, for example S199279817.",
    )
    input_group.add_argument(
        "--input-file",
        metavar="FILE",
        help="Text file with one alphanumeric ID per line.",
    )
    parser.add_argument(
        "--model-size",
        default="1.7B",
        choices=["0.6B", "1.7B"],
        help="Model size; 0.6B uses port 8011, 1.7B uses port 8021 (default: 1.7B).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override the API endpoint URL (default: derived from --model-size).",
    )
    parser.add_argument("--model", default="tts-1")
    parser.add_argument("--voice", default="default")
    parser.add_argument(
        "--response-format",
        default="wav",
        choices=["wav", "pcm", "mp3"],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of codec steps to generate (default: 2048).",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=2,
        help="Minimum number of codec steps before EOS is allowed (default: 2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling cutoff (default: 50).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling cutoff (default: 1.0).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty (default: 1.05).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=12,
        help="Streaming chunk size in codec steps (default: 12).",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help="Optional instruction prompt, for example dialect or style guidance.",
    )
    parser.add_argument(
        "--xvec-only",
        action="store_true",
        help="Use x-vector-only voice cloning instead of full ICL prompt.",
    )
    parser.add_argument(
        "--non-streaming-mode",
        action="store_true",
        help="Prefill the full target text before decode.",
    )
    parser.add_argument(
        "--no-append-silence",
        action="store_true",
        help="Disable the default trailing silence added to reference audio in ICL mode.",
    )
    parser.add_argument(
        "--parity-mode",
        action="store_true",
        help="Use parity streaming mode instead of the fast CUDA-graph path.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use greedy decoding.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of audio files to generate for each ID (default: 10).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/id_batch",
        help="Directory for generated audio files.",
    )
    parser.add_argument(
        "--with-prefix",
        action="store_true",
        help="Wrap spoken text as '身分证号码是，{id}，请问正确吗'.",
    )
    return parser.parse_args()


# Letters that have a fixed Mandarin pinyin spoken form.
# Letters not in this dict are kept as-is (uppercase).
LETTER_PINYIN: dict[str, str] = {
    "B": "bī",
    "D": "dī",
    "E": "yī",
    "G": "jū",
    "K": "kēi",
    "Y": "wāi",
    "Z": "lì",
}


def to_spoken_text(raw_id: str) -> str:
    stripped = raw_id.strip()
    # Separate leading letter prefix (e.g. 'H' in 'H124703542')
    i = 0
    while i < len(stripped) and stripped[i].isalpha():
        i += 1
    raw_prefix = stripped[:i].upper()
    rest = stripped[i:]

    # Apply pinyin substitution for the prefix letter(s)
    prefix = LETTER_PINYIN.get(raw_prefix, raw_prefix)

    # Convert digit characters to Chinese spoken form
    spoken_parts = [CHINESE_DIGITS.get(c, c.upper() if c.isalpha() else c) for c in rest]

    # Group digits into chunks of 3 separated by ", "
    groups = ["".join(spoken_parts[j:j + 3]) for j in range(0, len(spoken_parts), 3)]
    return prefix + " " + ", ".join(groups)


def synthesize_one(url: str, model: str, voice: str, fmt: str, text: str, args: argparse.Namespace) -> bytes:
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": fmt,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "do_sample": not args.no_sample,
        "repetition_penalty": args.repetition_penalty,
        "chunk_size": args.chunk_size,
        "xvec_only": args.xvec_only,
        "non_streaming_mode": args.non_streaming_mode,
        "append_silence": not args.no_append_silence,
        "parity_mode": args.parity_mode,
        "instruct": args.instruct,
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def load_ids(args: argparse.Namespace) -> list[str]:
    if args.single_id:
        return [args.single_id.strip()]
    input_path = Path(args.input_file)
    return [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]


_MODEL_PORTS: dict[str, int] = {"0.6B": 8011, "1.7B": 8021}
_MODEL_TAGS: dict[str, str] = {"0.6B": "0_6B", "1.7B": "1_7B"}


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be greater than 0.")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be greater than 0.")
    if args.min_new_tokens < 0:
        raise SystemExit("--min-new-tokens must be non-negative.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be greater than 0.")
    if args.top_k < 0:
        raise SystemExit("--top-k must be non-negative.")
    if not 0 < args.top_p <= 1.0:
        raise SystemExit("--top-p must be in the range (0, 1].")
    if args.repetition_penalty <= 0:
        raise SystemExit("--repetition-penalty must be greater than 0.")

    port = _MODEL_PORTS[args.model_size]
    url = args.url or f"http://127.0.0.1:{port}/v1/audio/speech"
    model_tag = _MODEL_TAGS[args.model_size]
    print(f"Model size: {args.model_size}  →  {url}")

    ids = load_ids(args)
    if not ids:
        raise SystemExit("No valid IDs found.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.response_format
    width = max(2, len(str(args.count)))
    generation_times_s: list[float] = []

    for raw_id in ids:
        spoken_text = to_spoken_text(raw_id)
        if args.with_prefix:
            spoken_text = f"身分证号码是，{spoken_text}，请问正确吗"
        prefix_tag = "_with_prefix" if args.with_prefix else ""
        print(f"{raw_id} -> {spoken_text}")
        for index in range(1, args.count + 1):
            t0 = time.perf_counter()
            audio_bytes = synthesize_one(
                url,
                args.model,
                args.voice,
                args.response_format,
                spoken_text,
                args,
            )
            filename = f"{raw_id}_{model_tag}{prefix_tag}_{index:0{width}d}.{suffix}"
            output_path = output_dir / filename
            output_path.write_bytes(audio_bytes)
            elapsed_s = time.perf_counter() - t0
            generation_times_s.append(elapsed_s)
            print(f"Saved {output_path}  generation_time={elapsed_s:.3f}s")

    if generation_times_s:
        avg_s = sum(generation_times_s) / len(generation_times_s)
        print(f"Average generation time: {avg_s:.3f}s over {len(generation_times_s)} file(s)")


if __name__ == "__main__":
    main()
