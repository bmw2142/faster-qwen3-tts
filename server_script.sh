#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

# MODEL_SIZE=0.6B -> port 8011
# MODEL_SIZE=1.7B -> port 8021
MODEL_SIZE="${MODEL_SIZE:-1.7B}"

case "$MODEL_SIZE" in
    0.6B)
        MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
        PORT="${PORT:-8011}"
        ;;
    1.7B)
        MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
        PORT="${PORT:-8021}"
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE (use 0.6B or 1.7B)" >&2
        exit 1
        ;;
esac

HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda:2}"
VOICES="${VOICES:-$SCRIPT_DIR/voices.json}"
# REF_AUDIO="${REF_AUDIO:-$SCRIPT_DIR/ref_audio/現在開始進行車牌語音合成 A, B, C, D, Q, U, V, W, Z.wav}"
# REF_TEXT="${REF_TEXT:-現在開始進行車牌語音合成 A, B, C, D, Q, U, V, W, Z}"
# REF_AUDIO="${REF_AUDIO:-$SCRIPT_DIR/ref_audio/你們這個火災保險是專門給我們這種小餐廳用的嗎？.wav}"
# REF_TEXT="${REF_TEXT:-你們這個火災保險是專門給我們這種小餐廳用的嗎？}"
LANGUAGE="${LANGUAGE:-Chinese}"

SERVER_ARGS=(
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --device "$DEVICE"
)

if [[ -f "$VOICES" ]]; then
    SERVER_ARGS+=(--voices "$VOICES")
else
    if [[ ! -f "$REF_AUDIO" ]]; then
        echo "Reference audio not found: $REF_AUDIO" >&2
        exit 1
    fi
    SERVER_ARGS+=(
        --ref-audio "$REF_AUDIO"
        --ref-text "$REF_TEXT"
        --language "$LANGUAGE"
    )
fi

exec python examples/openai_server.py "${SERVER_ARGS[@]}"
