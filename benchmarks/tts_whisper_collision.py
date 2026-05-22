#!/usr/bin/env python3
"""Benchmark Qwen3-TTS streaming latency while Whisper runs on the same GPU."""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402


DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_WHISPER_MODEL = "/data/jp-storage/model/whisper-large-v2"
DEFAULT_TTS_LIST = PROJECT_DIR / "collide_data" / "tts_data" / "tts.lst"
DEFAULT_ASR_DIR = PROJECT_DIR / "collide_data" / "whisper_data"
DEFAULT_VOICES = PROJECT_DIR / "voices.json"
DEFAULT_VOICE = "id_LQRMV_0423_04_traditional"


@dataclass
class JobMetric:
    run_id: str
    size: int
    offset: str
    asr_concurrency: int
    modality: str
    index: int
    input: str
    success: bool
    error: str | None
    scheduled_at_s: float
    lock_acquired_at_s: float | None
    first_audio_at_s: float | None
    ended_at_s: float
    queue_s: float | None
    inference_ttfa_s: float | None
    user_ttfa_s: float | None
    inference_total_s: float | None
    user_total_s: float
    audio_duration_s: float | None
    generated_audio_s: float | None
    chunks: int | None
    transcript: str | None


@dataclass
class ScenarioSummary:
    run_id: str
    size: int
    offset: str
    offset_label: str
    asr_concurrency: int
    total_time_s: float
    avg_user_total_s: float
    tts_user_ttfa_s: float | None
    tts_user_total_s: float | None
    asr_user_total_s: float | None
    tts_inference_ttfa_s: float | None
    tts_inference_total_s: float | None
    tts_queue_s: float | None
    error_count: int
    job_count: int


@dataclass
class PairwiseRecord:
    case: str
    run_id: str
    tts_index: int | None
    asr_index: int | None
    asr_index_2: int | None
    offset_s: float | None
    success: bool
    error: str | None
    wall_time_s: float
    tts_user_ttfa_s: float | None
    tts_user_total_s: float | None
    tts_inference_ttfa_s: float | None
    tts_inference_total_s: float | None
    asr_user_total_s: float | None
    asr_2_user_total_s: float | None
    asr_audio_duration_s: float | None
    asr_2_audio_duration_s: float | None
    tts_text: str | None
    asr_audio: str | None
    asr_audio_2: str | None


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_offsets(raw: str) -> list[str]:
    offsets = [part.strip() for part in raw.split(",") if part.strip()]
    if not offsets:
        raise argparse.ArgumentTypeError("expected at least one offset")
    for offset in offsets:
        if offset == "serial":
            continue
        try:
            value = float(offset)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid offset: {offset}") from exc
        if value < 0:
            raise argparse.ArgumentTypeError("offsets must be non-negative")
    return offsets


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure user-facing latency when streaming TTS and Whisper ASR collide.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument("--tts-list", type=Path, default=DEFAULT_TTS_LIST)
    parser.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR_DIR)
    parser.add_argument("--voices", type=Path, default=DEFAULT_VOICES)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "collide_results")
    parser.add_argument("--suite", choices=["batch", "pairwise"], default="batch")
    parser.add_argument("--sizes", type=parse_csv_ints, default=parse_csv_ints("1,2,3,4,5,6,7,10,20"))
    parser.add_argument("--offsets", type=parse_offsets, default=parse_offsets("serial,0,0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--pairwise-count", type=int, default=5)
    parser.add_argument("--pairwise-offsets", type=parse_csv_floats, default=parse_csv_floats("0.1,0.2,0.3,0.4,0.5"))
    parser.add_argument("--dual-asr-samples", type=int, default=5)
    parser.add_argument("--asr-concurrency", type=parse_csv_ints, default=parse_csv_ints("1"))
    parser.add_argument("--tts-concurrency", type=int, default=1)
    parser.add_argument("--worker-threads", type=int, default=4)
    parser.add_argument("--delayed-modality", choices=["asr", "tts"], default="asr")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--no-append-silence", action="store_true")
    parser.add_argument("--parity-mode", action="store_true")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--whisper-language", default="chinese")
    parser.add_argument("--whisper-task", default="transcribe")
    parser.add_argument("--whisper-max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--whisper-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if args.tts_concurrency != 1:
        raise SystemExit("Conservative TTS mode only supports --tts-concurrency 1.")
    if args.pairwise_count <= 0:
        raise SystemExit("--pairwise-count must be greater than 0.")
    if args.dual_asr_samples <= 0:
        raise SystemExit("--dual-asr-samples must be greater than 0.")
    if args.worker_threads <= 0:
        raise SystemExit("--worker-threads must be greater than 0.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be greater than 0.")
    if args.min_new_tokens < 0:
        raise SystemExit("--min-new-tokens must be non-negative.")
    return args


def set_cuda_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def rel_time(origin: float, value: float | None) -> float | None:
    if value is None:
        return None
    return value - origin


def mean_or_none(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return float(statistics.mean(clean)) if clean else None


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def offset_label(offset: str) -> str:
    if offset == "serial":
        return "序列"
    if float(offset) == 0:
        return "同時推理"
    return f"延遲 {float(offset):.1f} s"


def load_tts_texts(path: Path) -> list[str]:
    texts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise SystemExit(f"No TTS text found in {path}")
    return texts


def load_asr_paths(path: Path) -> list[Path]:
    paths = sorted(path.glob("*.wav"))
    if not paths:
        raise SystemExit(f"No wav files found in {path}")
    return paths


def audio_duration(path: Path) -> float:
    return float(sf.info(str(path)).duration)


def load_voice(voices_path: Path, voice_name: str) -> tuple[str, dict[str, Any]]:
    if not voices_path.exists():
        raise SystemExit(f"Voices file not found: {voices_path}")
    voices = json.loads(voices_path.read_text(encoding="utf-8"))
    if not voices:
        raise SystemExit(f"No voices configured in {voices_path}")
    resolved_name = voice_name if voice_name in voices else next(iter(voices))
    voice = dict(voices[resolved_name])
    ref_audio = Path(voice["ref_audio"])
    if not ref_audio.is_absolute():
        ref_audio = PROJECT_DIR / ref_audio
    voice["ref_audio"] = str(ref_audio)
    voice.setdefault("ref_text", "")
    voice.setdefault("language", "Chinese")
    return resolved_name, voice


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_audio_16k(path: Path) -> tuple[np.ndarray, float]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    original_duration = len(wav) / sr if sr else 0.0
    if sr != 16000:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav = torchaudio.functional.resample(wav_t, sr, 16000).squeeze(0).numpy()
    return wav, original_duration


class BenchmarkModels:
    def __init__(self, args: argparse.Namespace, voice: dict[str, Any]):
        self.args = args
        self.voice = voice
        self.tts_lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=args.worker_threads)

        print(f"Loading TTS model on {args.device}: {args.tts_model}", flush=True)
        set_cuda_device(args.device)
        self.tts = FasterQwen3TTS.from_pretrained(
            args.tts_model,
            device=args.device,
            dtype=torch_dtype(args.dtype),
            attn_implementation="eager",
            max_seq_len=2048,
        )
        sync_cuda(args.device)

        print(f"Loading Whisper model on {args.device}: {args.whisper_model}", flush=True)
        self.processor = WhisperProcessor.from_pretrained(args.whisper_model)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_model,
            torch_dtype=torch_dtype(args.whisper_dtype),
        ).to(args.device)
        self.whisper.eval()
        sync_cuda(args.device)

        self.whisper_generate_kwargs: dict[str, Any] = {}
        if args.whisper_language:
            self.whisper_generate_kwargs["language"] = args.whisper_language
        if args.whisper_task:
            self.whisper_generate_kwargs["task"] = args.whisper_task

    async def run_in_worker(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, fn, *args)

    def warmup(self, text: str, audio_path: Path) -> None:
        print("Warming up TTS streaming and Whisper...", flush=True)
        _ = self.run_tts_streaming_sync(text=text[:80], index=-1)
        _ = self.run_asr_sync(audio_path=audio_path, index=-1)
        print("Warmup done.", flush=True)

    def run_tts_streaming_sync(self, text: str, index: int) -> dict[str, Any]:
        del index
        set_cuda_device(self.args.device)
        sync_cuda(self.args.device)
        started_at = time.perf_counter()
        first_audio_at = None
        ended_at = started_at
        chunks = 0
        generated_audio_s = 0.0

        with self._thread_lock:
            generator = self.tts.generate_voice_clone_streaming(
                text=text,
                language=self.voice.get("language", self.args.language),
                ref_audio=self.voice["ref_audio"],
                ref_text=self.voice.get("ref_text", ""),
                max_new_tokens=self.args.max_new_tokens,
                min_new_tokens=self.args.min_new_tokens,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                do_sample=not self.args.no_sample,
                repetition_penalty=self.args.repetition_penalty,
                chunk_size=self.args.chunk_size,
                xvec_only=self.args.xvec_only,
                non_streaming_mode=self.args.non_streaming_mode,
                append_silence=not self.args.no_append_silence,
                parity_mode=self.args.parity_mode,
            )
            for audio_chunk, sr, _timing in generator:
                sync_cuda(self.args.device)
                now = time.perf_counter()
                if first_audio_at is None:
                    first_audio_at = now
                chunk = np.asarray(audio_chunk).squeeze()
                generated_audio_s += len(chunk) / sr if sr else 0.0
                chunks += 1
                ended_at = now

        sync_cuda(self.args.device)
        ended_at = time.perf_counter()
        return {
            "started_at": started_at,
            "first_audio_at": first_audio_at,
            "ended_at": ended_at,
            "chunks": chunks,
            "generated_audio_s": generated_audio_s,
        }

    def run_asr_sync(self, audio_path: Path, index: int) -> dict[str, Any]:
        del index
        set_cuda_device(self.args.device)
        wav, duration_s = load_audio_16k(audio_path)
        inputs = self.processor(
            wav,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            self.args.device,
            dtype=torch_dtype(self.args.whisper_dtype),
        )
        generate_kwargs = {
            **self.whisper_generate_kwargs,
            "max_new_tokens": self.args.whisper_max_new_tokens,
        }
        if "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs.attention_mask.to(self.args.device)

        sync_cuda(self.args.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.whisper.generate(input_features, **generate_kwargs)
        sync_cuda(self.args.device)
        ended_at = time.perf_counter()
        transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {
            "started_at": started_at,
            "ended_at": ended_at,
            "audio_duration_s": duration_s,
            "transcript": transcript,
        }


async def run_tts_job(
    models: BenchmarkModels,
    *,
    run_id: str,
    size: int,
    offset: str,
    asr_concurrency: int,
    index: int,
    text: str,
    scenario_origin: float,
    start_delay_s: float,
) -> JobMetric:
    if start_delay_s > 0:
        await asyncio.sleep(start_delay_s)
    scheduled_at = time.perf_counter()
    lock_acquired_at = None
    first_audio_at = None
    ended_at = scheduled_at
    error = None
    success = False
    generated_audio_s = None
    chunks = None
    try:
        async with models.tts_lock:
            lock_acquired_at = time.perf_counter()
            result = await models.run_in_worker(models.run_tts_streaming_sync, text, index)
        first_audio_at = result["first_audio_at"]
        ended_at = result["ended_at"]
        generated_audio_s = result["generated_audio_s"]
        chunks = result["chunks"]
        success = True
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
        ended_at = time.perf_counter()
        error = repr(exc)

    queue_s = lock_acquired_at - scheduled_at if lock_acquired_at is not None else None
    inference_ttfa_s = (
        first_audio_at - lock_acquired_at
        if first_audio_at is not None and lock_acquired_at is not None
        else None
    )
    user_ttfa_s = first_audio_at - scheduled_at if first_audio_at is not None else None
    inference_total_s = (
        ended_at - lock_acquired_at if lock_acquired_at is not None else None
    )
    user_total_s = ended_at - scheduled_at
    return JobMetric(
        run_id=run_id,
        size=size,
        offset=offset,
        asr_concurrency=asr_concurrency,
        modality="tts",
        index=index,
        input=text,
        success=success,
        error=error,
        scheduled_at_s=scheduled_at - scenario_origin,
        lock_acquired_at_s=rel_time(scenario_origin, lock_acquired_at),
        first_audio_at_s=rel_time(scenario_origin, first_audio_at),
        ended_at_s=ended_at - scenario_origin,
        queue_s=queue_s,
        inference_ttfa_s=inference_ttfa_s,
        user_ttfa_s=user_ttfa_s,
        inference_total_s=inference_total_s,
        user_total_s=user_total_s,
        audio_duration_s=None,
        generated_audio_s=generated_audio_s,
        chunks=chunks,
        transcript=None,
    )


async def run_asr_job(
    models: BenchmarkModels,
    semaphore: asyncio.Semaphore,
    *,
    run_id: str,
    size: int,
    offset: str,
    asr_concurrency: int,
    index: int,
    audio_path: Path,
    scenario_origin: float,
    start_delay_s: float,
) -> JobMetric:
    if start_delay_s > 0:
        await asyncio.sleep(start_delay_s)
    scheduled_at = time.perf_counter()
    started_at = None
    ended_at = scheduled_at
    error = None
    success = False
    duration_s = None
    transcript = None
    try:
        async with semaphore:
            started_at = time.perf_counter()
            result = await models.run_in_worker(models.run_asr_sync, audio_path, index)
        ended_at = result["ended_at"]
        duration_s = result["audio_duration_s"]
        transcript = result["transcript"]
        success = True
    except Exception as exc:  # noqa: BLE001 - benchmark should keep going.
        ended_at = time.perf_counter()
        error = repr(exc)

    return JobMetric(
        run_id=run_id,
        size=size,
        offset=offset,
        asr_concurrency=asr_concurrency,
        modality="asr",
        index=index,
        input=str(audio_path),
        success=success,
        error=error,
        scheduled_at_s=scheduled_at - scenario_origin,
        lock_acquired_at_s=rel_time(scenario_origin, started_at),
        first_audio_at_s=None,
        ended_at_s=ended_at - scenario_origin,
        queue_s=started_at - scheduled_at if started_at is not None else None,
        inference_ttfa_s=None,
        user_ttfa_s=None,
        inference_total_s=ended_at - started_at if started_at is not None else None,
        user_total_s=ended_at - scheduled_at,
        audio_duration_s=duration_s,
        generated_audio_s=None,
        chunks=None,
        transcript=transcript,
    )


async def run_scenario(
    models: BenchmarkModels,
    *,
    run_id: str,
    size: int,
    offset: str,
    asr_concurrency: int,
    tts_texts: list[str],
    asr_paths: list[Path],
    delayed_modality: str,
) -> tuple[ScenarioSummary, list[JobMetric]]:
    scenario_origin = time.perf_counter()
    semaphore = asyncio.Semaphore(asr_concurrency)
    jobs: list[JobMetric] = []

    if offset == "serial":
        for index, text in enumerate(tts_texts[:size]):
            jobs.append(
                await run_tts_job(
                    models,
                    run_id=run_id,
                    size=size,
                    offset=offset,
                    asr_concurrency=asr_concurrency,
                    index=index,
                    text=text,
                    scenario_origin=scenario_origin,
                    start_delay_s=0.0,
                )
            )
        for index, audio_path in enumerate(asr_paths[:size]):
            jobs.append(
                await run_asr_job(
                    models,
                    semaphore,
                    run_id=run_id,
                    size=size,
                    offset=offset,
                    asr_concurrency=asr_concurrency,
                    index=index,
                    audio_path=audio_path,
                    scenario_origin=scenario_origin,
                    start_delay_s=0.0,
                )
            )
    else:
        delay = float(offset)
        tts_delay = delay if delayed_modality == "tts" else 0.0
        asr_delay = delay if delayed_modality == "asr" else 0.0
        tasks = [
            asyncio.create_task(
                run_tts_job(
                    models,
                    run_id=run_id,
                    size=size,
                    offset=offset,
                    asr_concurrency=asr_concurrency,
                    index=index,
                    text=text,
                    scenario_origin=scenario_origin,
                    start_delay_s=tts_delay,
                )
            )
            for index, text in enumerate(tts_texts[:size])
        ]
        tasks.extend(
            asyncio.create_task(
                run_asr_job(
                    models,
                    semaphore,
                    run_id=run_id,
                    size=size,
                    offset=offset,
                    asr_concurrency=asr_concurrency,
                    index=index,
                    audio_path=audio_path,
                    scenario_origin=scenario_origin,
                    start_delay_s=asr_delay,
                )
            )
            for index, audio_path in enumerate(asr_paths[:size])
        )
        jobs = list(await asyncio.gather(*tasks))

    scenario_end = max((job.ended_at_s for job in jobs), default=0.0)
    summary = summarize_scenario(run_id, size, offset, asr_concurrency, scenario_end, jobs)
    return summary, jobs


def summarize_scenario(
    run_id: str,
    size: int,
    offset: str,
    asr_concurrency: int,
    total_time_s: float,
    jobs: list[JobMetric],
) -> ScenarioSummary:
    tts_jobs = [job for job in jobs if job.modality == "tts" and job.success]
    asr_jobs = [job for job in jobs if job.modality == "asr" and job.success]
    successful_jobs = [job for job in jobs if job.success]
    return ScenarioSummary(
        run_id=run_id,
        size=size,
        offset=offset,
        offset_label=offset_label(offset),
        asr_concurrency=asr_concurrency,
        total_time_s=total_time_s,
        avg_user_total_s=mean_or_none([job.user_total_s for job in successful_jobs]) or 0.0,
        tts_user_ttfa_s=mean_or_none([job.user_ttfa_s for job in tts_jobs if job.user_ttfa_s is not None]),
        tts_user_total_s=mean_or_none([job.user_total_s for job in tts_jobs]),
        asr_user_total_s=mean_or_none([job.user_total_s for job in asr_jobs]),
        tts_inference_ttfa_s=mean_or_none(
            [job.inference_ttfa_s for job in tts_jobs if job.inference_ttfa_s is not None]
        ),
        tts_inference_total_s=mean_or_none(
            [job.inference_total_s for job in tts_jobs if job.inference_total_s is not None]
        ),
        tts_queue_s=mean_or_none([job.queue_s for job in tts_jobs if job.queue_s is not None]),
        error_count=len([job for job in jobs if not job.success]),
        job_count=len(jobs),
    )


def audio_stats_for_sizes(asr_paths: list[Path], sizes: list[int]) -> list[dict[str, float]]:
    durations = [audio_duration(path) for path in asr_paths]
    rows = []
    for size in sizes:
        subset = durations[:size]
        rows.append(
            {
                "size": size,
                "avg": float(statistics.mean(subset)),
                "max": float(max(subset)),
                "min": float(min(subset)),
                "std": float(statistics.stdev(subset)) if len(subset) > 1 else 0.0,
            }
        )
    return rows


def pivot_table(
    summaries: list[ScenarioSummary],
    *,
    asr_concurrency: int,
    sizes: list[int],
    offsets: list[str],
    metric: str,
) -> list[list[str]]:
    lookup = {
        (summary.size, summary.offset): getattr(summary, metric)
        for summary in summaries
        if summary.asr_concurrency == asr_concurrency
    }
    rows = [["數量", *[offset_label(offset) for offset in offsets]]]
    for size in sizes:
        rows.append([str(size), *[fmt(lookup.get((size, offset))) for offset in offsets]])
    return rows


def markdown_table(headers_and_rows: list[list[str]]) -> str:
    if not headers_and_rows:
        return ""
    header = headers_and_rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in headers_and_rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(
    summaries: list[ScenarioSummary],
    audio_rows: list[dict[str, float]],
    *,
    sizes: list[int],
    offsets: list[str],
    asr_concurrency_values: list[int],
    args: argparse.Namespace,
    voice_name: str,
) -> str:
    lines = [
        "# TTS + Whisper Collision Benchmark",
        "",
        f"- Device: `{args.device}`",
        f"- TTS model: `{args.tts_model}`",
        f"- Whisper model: `{args.whisper_model}`",
        f"- Voice: `{voice_name}`",
        f"- TTS chunk size: `{args.chunk_size}`",
        f"- Delayed modality: `{args.delayed_modality}`",
        "",
        "## 音檔長度(s)",
        "",
    ]
    audio_table = [["數量", "平均", "最長", "最短", "標準差"]]
    for row in audio_rows:
        audio_table.append(
            [
                str(int(row["size"])),
                fmt(row["avg"]),
                fmt(row["max"]),
                fmt(row["min"]),
                fmt(row["std"]),
            ]
        )
    lines.append(markdown_table(audio_table))

    metric_titles = [
        ("total_time_s", "total time(s)"),
        ("avg_user_total_s", "average user total time(s)"),
        ("tts_user_ttfa_s", "TTS user TTFA(s)"),
        ("tts_user_total_s", "TTS user total time(s)"),
        ("asr_user_total_s", "ASR user total time(s)"),
    ]
    for asr_concurrency in asr_concurrency_values:
        lines.extend(["", f"## ASR concurrency = {asr_concurrency}", ""])
        for metric, title in metric_titles:
            lines.extend(
                [
                    f"### {title}",
                    "",
                    markdown_table(
                        pivot_table(
                            summaries,
                            asr_concurrency=asr_concurrency,
                            sizes=sizes,
                            offsets=offsets,
                            metric=metric,
                        )
                    ),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    output_dir: Path,
    summaries: list[ScenarioSummary],
    jobs: list[JobMetric],
    report: str,
    audio_rows: list[dict[str, float]],
    args: argparse.Namespace,
    voice_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "voice": voice_name,
                "audio_stats": audio_rows,
                "summaries": [asdict(summary) for summary in summaries],
                "jobs": [asdict(job) for job in jobs],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))

    with (output_dir / "job_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(jobs[0]).keys()))
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))


def pairwise_record_from_jobs(
    *,
    case: str,
    run_id: str,
    jobs: list[JobMetric],
    tts_index: int | None,
    asr_index: int | None,
    asr_index_2: int | None,
    offset_s: float | None,
    tts_text: str | None,
    asr_audio: Path | None,
    asr_audio_2: Path | None,
) -> PairwiseRecord:
    tts_jobs = [job for job in jobs if job.modality == "tts"]
    asr_jobs = [job for job in jobs if job.modality == "asr"]
    errors = [job.error for job in jobs if job.error]
    asr_1 = asr_jobs[0] if len(asr_jobs) >= 1 else None
    asr_2 = asr_jobs[1] if len(asr_jobs) >= 2 else None
    tts = tts_jobs[0] if tts_jobs else None
    return PairwiseRecord(
        case=case,
        run_id=run_id,
        tts_index=tts_index,
        asr_index=asr_index,
        asr_index_2=asr_index_2,
        offset_s=offset_s,
        success=all(job.success for job in jobs),
        error="; ".join(errors) if errors else None,
        wall_time_s=max((job.ended_at_s for job in jobs), default=0.0),
        tts_user_ttfa_s=tts.user_ttfa_s if tts else None,
        tts_user_total_s=tts.user_total_s if tts else None,
        tts_inference_ttfa_s=tts.inference_ttfa_s if tts else None,
        tts_inference_total_s=tts.inference_total_s if tts else None,
        asr_user_total_s=asr_1.user_total_s if asr_1 else None,
        asr_2_user_total_s=asr_2.user_total_s if asr_2 else None,
        asr_audio_duration_s=asr_1.audio_duration_s if asr_1 else None,
        asr_2_audio_duration_s=asr_2.audio_duration_s if asr_2 else None,
        tts_text=tts_text,
        asr_audio=str(asr_audio) if asr_audio else None,
        asr_audio_2=str(asr_audio_2) if asr_audio_2 else None,
    )


async def run_pairwise_tts_alone(
    models: BenchmarkModels,
    *,
    run_id: str,
    index: int,
    text: str,
) -> tuple[PairwiseRecord, list[JobMetric]]:
    scenario_origin = time.perf_counter()
    job = await run_tts_job(
        models,
        run_id=run_id,
        size=1,
        offset="tts_alone",
        asr_concurrency=0,
        index=index,
        text=text,
        scenario_origin=scenario_origin,
        start_delay_s=0.0,
    )
    jobs = [job]
    return (
        pairwise_record_from_jobs(
            case="1_tts_alone",
            run_id=run_id,
            jobs=jobs,
            tts_index=index,
            asr_index=None,
            asr_index_2=None,
            offset_s=None,
            tts_text=text,
            asr_audio=None,
            asr_audio_2=None,
        ),
        jobs,
    )


async def run_pairwise_asr_alone(
    models: BenchmarkModels,
    *,
    run_id: str,
    index: int,
    audio_path: Path,
) -> tuple[PairwiseRecord, list[JobMetric]]:
    scenario_origin = time.perf_counter()
    semaphore = asyncio.Semaphore(1)
    job = await run_asr_job(
        models,
        semaphore,
        run_id=run_id,
        size=1,
        offset="asr_alone",
        asr_concurrency=1,
        index=index,
        audio_path=audio_path,
        scenario_origin=scenario_origin,
        start_delay_s=0.0,
    )
    jobs = [job]
    return (
        pairwise_record_from_jobs(
            case="2_asr_alone",
            run_id=run_id,
            jobs=jobs,
            tts_index=None,
            asr_index=index,
            asr_index_2=None,
            offset_s=None,
            tts_text=None,
            asr_audio=audio_path,
            asr_audio_2=None,
        ),
        jobs,
    )


async def run_pairwise_collision(
    models: BenchmarkModels,
    *,
    case: str,
    run_id: str,
    tts_index: int,
    text: str,
    asr_index: int,
    audio_path: Path,
    offset_s: float,
) -> tuple[PairwiseRecord, list[JobMetric]]:
    scenario_origin = time.perf_counter()
    semaphore = asyncio.Semaphore(1)
    jobs = list(
        await asyncio.gather(
            run_tts_job(
                models,
                run_id=run_id,
                size=1,
                offset=str(offset_s),
                asr_concurrency=1,
                index=tts_index,
                text=text,
                scenario_origin=scenario_origin,
                start_delay_s=0.0,
            ),
            run_asr_job(
                models,
                semaphore,
                run_id=run_id,
                size=1,
                offset=str(offset_s),
                asr_concurrency=1,
                index=asr_index,
                audio_path=audio_path,
                scenario_origin=scenario_origin,
                start_delay_s=offset_s,
            ),
        )
    )
    return (
        pairwise_record_from_jobs(
            case=case,
            run_id=run_id,
            jobs=jobs,
            tts_index=tts_index,
            asr_index=asr_index,
            asr_index_2=None,
            offset_s=offset_s,
            tts_text=text,
            asr_audio=audio_path,
            asr_audio_2=None,
        ),
        jobs,
    )


async def run_pairwise_dual_asr(
    models: BenchmarkModels,
    *,
    run_id: str,
    tts_index: int,
    text: str,
    asr_index: int,
    audio_path: Path,
    asr_index_2: int,
    audio_path_2: Path,
) -> tuple[PairwiseRecord, list[JobMetric]]:
    scenario_origin = time.perf_counter()
    semaphore = asyncio.Semaphore(2)
    jobs = list(
        await asyncio.gather(
            run_tts_job(
                models,
                run_id=run_id,
                size=1,
                offset="dual_asr",
                asr_concurrency=2,
                index=tts_index,
                text=text,
                scenario_origin=scenario_origin,
                start_delay_s=0.0,
            ),
            run_asr_job(
                models,
                semaphore,
                run_id=run_id,
                size=1,
                offset="dual_asr",
                asr_concurrency=2,
                index=asr_index,
                audio_path=audio_path,
                scenario_origin=scenario_origin,
                start_delay_s=0.0,
            ),
            run_asr_job(
                models,
                semaphore,
                run_id=run_id,
                size=1,
                offset="dual_asr",
                asr_concurrency=2,
                index=asr_index_2,
                audio_path=audio_path_2,
                scenario_origin=scenario_origin,
                start_delay_s=0.0,
            ),
        )
    )
    return (
        pairwise_record_from_jobs(
            case="5_tts_with_two_asr",
            run_id=run_id,
            jobs=jobs,
            tts_index=tts_index,
            asr_index=asr_index,
            asr_index_2=asr_index_2,
            offset_s=0.0,
            tts_text=text,
            asr_audio=audio_path,
            asr_audio_2=audio_path_2,
        ),
        jobs,
    )


def summarize_pairwise_case(records: list[PairwiseRecord]) -> list[list[str]]:
    rows = [["case", "runs", "success", "wall avg", "TTS TTFA avg", "TTS total avg", "ASR avg", "ASR2 avg"]]
    for case in [
        "1_tts_alone",
        "2_asr_alone",
        "3_tts_with_asr_simultaneous",
        "4_tts_with_asr_offset",
        "5_tts_with_two_asr",
    ]:
        subset = [record for record in records if record.case == case]
        if not subset:
            continue
        rows.append(
            [
                case,
                str(len(subset)),
                str(len([record for record in subset if record.success])),
                fmt(mean_or_none([record.wall_time_s for record in subset])),
                fmt(mean_or_none([record.tts_user_ttfa_s for record in subset if record.tts_user_ttfa_s is not None])),
                fmt(mean_or_none([record.tts_user_total_s for record in subset if record.tts_user_total_s is not None])),
                fmt(mean_or_none([record.asr_user_total_s for record in subset if record.asr_user_total_s is not None])),
                fmt(mean_or_none([record.asr_2_user_total_s for record in subset if record.asr_2_user_total_s is not None])),
            ]
        )
    return rows


def build_pairwise_report(
    records: list[PairwiseRecord],
    audio_rows: list[dict[str, float]],
    *,
    args: argparse.Namespace,
    voice_name: str,
    tts_count: int,
    asr_count: int,
) -> str:
    lines = [
        "# TTS + Whisper Pairwise Collision Benchmark",
        "",
        f"- Device: `{args.device}`",
        f"- TTS model: `{args.tts_model}`",
        f"- Whisper model: `{args.whisper_model}`",
        f"- Voice: `{voice_name}`",
        f"- TTS chunk size: `{args.chunk_size}`",
        f"- TTS scripts: `{tts_count}`",
        f"- ASR audio files: `{asr_count}`",
        f"- Worker threads: `{args.worker_threads}`",
        "",
        "## Summary",
        "",
        markdown_table(summarize_pairwise_case(records)),
        "",
        "## 音檔長度(s)",
        "",
    ]
    audio_table = [["數量", "平均", "最長", "最短", "標準差"]]
    for row in audio_rows:
        audio_table.append(
            [str(int(row["size"])), fmt(row["avg"]), fmt(row["max"]), fmt(row["min"]), fmt(row["std"])]
        )
    lines.append(markdown_table(audio_table))

    offset_records = [record for record in records if record.case == "4_tts_with_asr_offset"]
    if offset_records:
        lines.extend(["", "## Offset Average", ""])
        rows = [["offset(s)", "runs", "wall avg", "TTS TTFA avg", "TTS total avg", "ASR avg"]]
        for offset in args.pairwise_offsets:
            subset = [record for record in offset_records if record.offset_s == offset]
            rows.append(
                [
                    fmt(offset, 1),
                    str(len(subset)),
                    fmt(mean_or_none([record.wall_time_s for record in subset])),
                    fmt(mean_or_none([record.tts_user_ttfa_s for record in subset if record.tts_user_ttfa_s is not None])),
                    fmt(mean_or_none([record.tts_user_total_s for record in subset if record.tts_user_total_s is not None])),
                    fmt(mean_or_none([record.asr_user_total_s for record in subset if record.asr_user_total_s is not None])),
                ]
            )
        lines.append(markdown_table(rows))

    return "\n".join(lines).rstrip() + "\n"


def write_pairwise_outputs(
    output_dir: Path,
    records: list[PairwiseRecord],
    jobs: list[JobMetric],
    report: str,
    audio_rows: list[dict[str, float]],
    args: argparse.Namespace,
    voice_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairwise_report.md").write_text(report, encoding="utf-8")
    with (output_dir / "pairwise_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "voice": voice_name,
                "audio_stats": audio_rows,
                "records": [asdict(record) for record in records],
                "jobs": [asdict(job) for job in jobs],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (output_dir / "pairwise_records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    with (output_dir / "pairwise_job_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(jobs[0]).keys()))
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))


async def run_pairwise_suite(args: argparse.Namespace) -> Path:
    tts_texts = load_tts_texts(args.tts_list)
    asr_paths = load_asr_paths(args.asr_dir)
    count = args.pairwise_count
    if len(tts_texts) < count:
        raise SystemExit(f"{args.tts_list} has {len(tts_texts)} texts, but --pairwise-count is {count}.")
    if len(asr_paths) < count:
        raise SystemExit(f"{args.asr_dir} has {len(asr_paths)} wav files, but --pairwise-count is {count}.")
    tts_texts = tts_texts[:count]
    asr_paths = asr_paths[:count]

    voice_name, voice = load_voice(args.voices, args.voice)
    audio_rows = audio_stats_for_sizes(asr_paths, [count])
    models = BenchmarkModels(args, voice)
    if not args.no_warmup:
        models.warmup(tts_texts[0], asr_paths[0])

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_stamp
    records: list[PairwiseRecord] = []
    all_jobs: list[JobMetric] = []

    planned = count + count + (count * count) + (count * count * len(args.pairwise_offsets)) + args.dual_asr_samples
    current = 0

    for tts_index, text in enumerate(tts_texts):
        current += 1
        print(f"[{current}/{planned}] case=1_tts_alone tts={tts_index}", flush=True)
        record, jobs = await run_pairwise_tts_alone(
            models,
            run_id=f"tts_alone_t{tts_index}",
            index=tts_index,
            text=text,
        )
        records.append(record)
        all_jobs.extend(jobs)

    for asr_index, audio_path in enumerate(asr_paths):
        current += 1
        print(f"[{current}/{planned}] case=2_asr_alone asr={asr_index}", flush=True)
        record, jobs = await run_pairwise_asr_alone(
            models,
            run_id=f"asr_alone_a{asr_index}",
            index=asr_index,
            audio_path=audio_path,
        )
        records.append(record)
        all_jobs.extend(jobs)

    for tts_index, text in enumerate(tts_texts):
        for asr_index, audio_path in enumerate(asr_paths):
            current += 1
            print(
                f"[{current}/{planned}] case=3_tts_with_asr_simultaneous "
                f"tts={tts_index} asr={asr_index}",
                flush=True,
            )
            record, jobs = await run_pairwise_collision(
                models,
                case="3_tts_with_asr_simultaneous",
                run_id=f"sim_t{tts_index}_a{asr_index}",
                tts_index=tts_index,
                text=text,
                asr_index=asr_index,
                audio_path=audio_path,
                offset_s=0.0,
            )
            records.append(record)
            all_jobs.extend(jobs)

    for offset in args.pairwise_offsets:
        for tts_index, text in enumerate(tts_texts):
            for asr_index, audio_path in enumerate(asr_paths):
                current += 1
                print(
                    f"[{current}/{planned}] case=4_tts_with_asr_offset "
                    f"offset={offset:.1f}s tts={tts_index} asr={asr_index}",
                    flush=True,
                )
                record, jobs = await run_pairwise_collision(
                    models,
                    case="4_tts_with_asr_offset",
                    run_id=f"off{offset:.1f}_t{tts_index}_a{asr_index}".replace(".", "p"),
                    tts_index=tts_index,
                    text=text,
                    asr_index=asr_index,
                    audio_path=audio_path,
                    offset_s=offset,
                )
                records.append(record)
                all_jobs.extend(jobs)

    for sample_index in range(args.dual_asr_samples):
        tts_index = sample_index % count
        asr_index = sample_index % count
        asr_index_2 = (sample_index + 1) % count
        current += 1
        print(
            f"[{current}/{planned}] case=5_tts_with_two_asr "
            f"tts={tts_index} asr={asr_index},{asr_index_2}",
            flush=True,
        )
        record, jobs = await run_pairwise_dual_asr(
            models,
            run_id=f"dual_t{tts_index}_a{asr_index}_{asr_index_2}",
            tts_index=tts_index,
            text=tts_texts[tts_index],
            asr_index=asr_index,
            audio_path=asr_paths[asr_index],
            asr_index_2=asr_index_2,
            audio_path_2=asr_paths[asr_index_2],
        )
        records.append(record)
        all_jobs.extend(jobs)

    report = build_pairwise_report(
        records,
        audio_rows,
        args=args,
        voice_name=voice_name,
        tts_count=count,
        asr_count=count,
    )
    write_pairwise_outputs(output_dir, records, all_jobs, report, audio_rows, args, voice_name)
    print("\n" + report, flush=True)
    return output_dir


async def run_all(args: argparse.Namespace) -> Path:
    if args.suite == "pairwise":
        return await run_pairwise_suite(args)

    tts_texts = load_tts_texts(args.tts_list)
    asr_paths = load_asr_paths(args.asr_dir)
    max_size = max(args.sizes)
    if len(tts_texts) < max_size:
        raise SystemExit(f"{args.tts_list} has {len(tts_texts)} texts, but max size is {max_size}.")
    if len(asr_paths) < max_size:
        raise SystemExit(f"{args.asr_dir} has {len(asr_paths)} wav files, but max size is {max_size}.")

    voice_name, voice = load_voice(args.voices, args.voice)
    audio_rows = audio_stats_for_sizes(asr_paths, args.sizes)
    models = BenchmarkModels(args, voice)
    if not args.no_warmup:
        models.warmup(tts_texts[0], asr_paths[0])

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_stamp
    summaries: list[ScenarioSummary] = []
    all_jobs: list[JobMetric] = []

    total_scenarios = len(args.asr_concurrency) * len(args.sizes) * len(args.offsets)
    scenario_number = 0
    for asr_concurrency in args.asr_concurrency:
        for size in args.sizes:
            for offset in args.offsets:
                scenario_number += 1
                run_id = f"c{asr_concurrency}_n{size}_{offset.replace('.', 'p')}"
                print(
                    f"[{scenario_number}/{total_scenarios}] "
                    f"size={size} offset={offset_label(offset)} asr_concurrency={asr_concurrency}",
                    flush=True,
                )
                summary, jobs = await run_scenario(
                    models,
                    run_id=run_id,
                    size=size,
                    offset=offset,
                    asr_concurrency=asr_concurrency,
                    tts_texts=tts_texts,
                    asr_paths=asr_paths,
                    delayed_modality=args.delayed_modality,
                )
                summaries.append(summary)
                all_jobs.extend(jobs)
                if summary.error_count:
                    print(f"  errors: {summary.error_count}/{summary.job_count}", flush=True)
                print(
                    f"  total={summary.total_time_s:.2f}s "
                    f"avg_user={summary.avg_user_total_s:.2f}s "
                    f"tts_ttfa={fmt(summary.tts_user_ttfa_s)}s",
                    flush=True,
                )

    report = build_report(
        summaries,
        audio_rows,
        sizes=args.sizes,
        offsets=args.offsets,
        asr_concurrency_values=args.asr_concurrency,
        args=args,
        voice_name=voice_name,
    )
    write_outputs(output_dir, summaries, all_jobs, report, audio_rows, args, voice_name)
    print("\n" + report, flush=True)
    return output_dir


def main() -> None:
    args = parse_args()
    output_dir = asyncio.run(run_all(args))
    print(f"Saved benchmark outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    main()
