#!/usr/bin/env bash
# Self-contained runner for the doomai repo.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
VENV="$ROOT/.venv"

MODE="hybrid-demos"
TARGET=""
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
doomai runner

Usage:
  bash scripts/run.sh [mode] [options]

Modes:
  --collect-demos
      Play the game yourself and save demos into demos/human by default.

  --bc-pretrain
      Behavior-clone a DQN initializer from saved demos.

  --hybrid-demos
      Behavior-clone from demos, run DQN warm-up, then PPO fine-tuning.

  --train
      Fresh PPO run without human demos.

  --dqn
      Fresh DQN run without human demos.

  --hybrid
      Fresh DQN warm-up followed by PPO, without human demos.

  --eval MODEL_ONNX
      Evaluate an exported model.

  --view RUN_OR_MODEL_OR_CKPT
      Record a replay gif/mp4.

Examples:
  bash scripts/run.sh --collect-demos --episodes 5
  bash scripts/run.sh --bc-pretrain --demo-dir demos/human --bc-epochs 8
  bash scripts/run.sh --hybrid-demos --demo-dir demos/human --total-steps 1000000 --hybrid-dqn-steps 200000
  bash scripts/run.sh --hybrid --total-steps 1000000 --hybrid-dqn-steps 200000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect-demos) MODE="collect-demos"; shift ;;
        --bc-pretrain) MODE="bc-pretrain"; shift ;;
        --hybrid-demos) MODE="hybrid-demos"; shift ;;
        --train) MODE="train"; shift ;;
        --dqn) MODE="dqn"; shift ;;
        --hybrid) MODE="hybrid"; shift ;;
        --eval) MODE="eval"; EXTRA_ARGS+=("--submission" "$2"); shift 2 ;;
        --view) MODE="view"; TARGET="$2"; shift 2 ;;
        -h|--help)
            usage
            exit 0 ;;
        --keep-noop|--drop-noop|--balance-actions|--no-eval)
            EXTRA_ARGS+=("$1"); shift ;;
        *)
            EXTRA_ARGS+=("$1")
            if [[ $# -ge 2 && "${2:-}" != --* ]]; then
                EXTRA_ARGS+=("$2")
                shift 2
            else
                shift
            fi
            ;;
    esac
done

if [[ -x "$VENV/bin/python" ]]; then
    PY="$VENV/bin/python"
elif [[ -x "$ROOT/../.venv/bin/python" ]]; then
    PY="$ROOT/../.venv/bin/python"
else
    PY="python3"
fi

case "$MODE" in
    collect-demos)
        PYTHONPATH="$ROOT" "$PY" -m utility.cli --collect-demos "${EXTRA_ARGS[@]}"
        ;;
    bc-pretrain)
        PYTHONPATH="$ROOT" "$PY" -m utility.cli --bc-pretrain "${EXTRA_ARGS[@]}"
        ;;
    hybrid-demos)
        PYTHONPATH="$ROOT" "$PY" -m utility.cli --hybrid-from-demos "${EXTRA_ARGS[@]}"
        ;;
    train)
        PYTHONPATH="$ROOT" "$PY" -m core.rl "${EXTRA_ARGS[@]}"
        ;;
    dqn)
        PYTHONPATH="$ROOT" "$PY" -m core.rl --algo dqn "${EXTRA_ARGS[@]}"
        ;;
    hybrid)
        PYTHONPATH="$ROOT" "$PY" -m core.rl --algo hybrid "${EXTRA_ARGS[@]}"
        ;;
    eval)
        PYTHONPATH="$ROOT" "$PY" -m utility.eval "${EXTRA_ARGS[@]}"
        ;;
    view)
        PYTHONPATH="$ROOT" "$PY" -m utility.view "$TARGET" "${EXTRA_ARGS[@]}"
        ;;
esac
