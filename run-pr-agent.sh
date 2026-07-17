#!/bin/bash
# Local PR Agent Runner
# Usage: ./run-pr-agent.sh <pr-url> <command>
#   command: review, describe, improve, ask "question"
#
# Example:
#   ./run-pr-agent.sh https://github.com/lemonade-sdk/lemonade/pull/2448 review

set -e

PR_URL="${1}"
COMMAND="${2:-review}"
ARGS="${@:3}"

if [ -z "$PR_URL" ]; then
    echo "Usage: $0 <pr-url> [command] [args...]"
    echo "  command: review (default), describe, improve, ask"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv-pr-agent"

if [ ! -d "$VENV" ]; then
    echo "Error: Virtual environment not found at $VENV"
    exit 1
fi

# Source the venv
source "$VENV/bin/activate"

# Run the PR agent with local Ollama model
CONFIG__MODEL="ollama_chat/gpt-oss:20b" \
CONFIG__FALLBACK_MODELS='["ollama_chat/qwen3.5:9b"]' \
CONFIG__MODEL_TURBO="ollama_chat/gpt-oss:20b" \
CONFIG__CUSTOM_MODEL_MAX_TOKENS=32000 \
CONFIG__VERBOSITY_LEVEL=0 \
CONFIG__PUBLISH_OUTPUT=false \
CONFIG__SKIP_KEYS='[]' \
CONFIG__USE_REPO_SETTINGS_FILE=false \
CONFIG__USE_WIKI_SETTINGS_FILE=false \
CONFIG__USE_GLOBAL_SETTINGS_FILE=false \
CONFIG__AI_TIMEOUT=300 \
OLLAMA_API_BASE="http://127.0.0.1:11434" \
pr-agent --pr_url="$PR_URL" $COMMAND $ARGS

echo ""
echo "✅ Done! (output was not published to GitHub since publish_output=false)"
echo "   To publish to GitHub, set: CONFIG__PUBLISH_OUTPUT=true"
