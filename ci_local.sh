#!/usr/bin/env bash
# Local CI — mirrors the GitHub CI pipeline so we don't burn maintainer credits.
# Usage:  bash ci_local.sh [--build-only] [--test-model model_name]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
START_TS=$(date +%s)
PASS=true

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }
step()   { blue "━━━ $* ━━━"; }

cleanup() {
    local rc=$?
    local dur=$(( $(date +%s) - START_TS ))
    if $PASS; then green "✅ CI LOCAL PASSED (${dur}s)"; else red "❌ CI LOCAL FAILED (${dur}s)"; fi
    exit $rc
}
trap cleanup EXIT

# ── Config ──
BUILD_ONLY=false
TEST_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only) BUILD_ONLY=true; shift ;;
        --test-model) TEST_MODEL="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── Step 1: Clean build ──
step "1/4: Clean CMake configure + build"
rm -rf "${BUILD_DIR}" 2>/dev/null || true
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_CLANG_TIDY= \
    -DCLANG_TIDY_EXE=CLANG_TIDY_EXE-NOTFOUND \
    -DMINJA_EXAMPLE_ENABLED=OFF \
    -DMLX_BUILD_ROCM=OFF \
    -DMLX_LM_BUILD_TESTS=ON \
    -DMLX_LM_BUILD_EXAMPLES=ON \
    2>&1 | tail -5

cmake --build . -j "$(nproc)" 2>&1 | tail -5
green "  Build OK"

# ── Step 2: Unit tests ──
step "2/4: Unit tests"
ctest --test-dir tests \
    -R "test_types|test_config|test_generate|test_kv_cache|test_chat_template|test_rope_utils|test_bitnet" \
    --output-on-failure --timeout 120 2>&1 | tail -3
green "  All unit tests passed"

# ── Step 3: Server binary smoke test ──
step "3/4: Server binary check"
if [[ -x "${BUILD_DIR}/server" ]]; then
    ldd "${BUILD_DIR}/server" 2>&1 | grep "not found" && { red "Missing libs!"; PASS=false; exit 1; }
    green "  All dependencies resolved"
else
    red "  server binary not found"
    PASS=false; exit 1
fi

if $BUILD_ONLY; then green "  --build-only, skipping model test"; exit 0; fi

# ── Step 4: Optional model smoke test ──
MODEL="${TEST_MODEL:-mlx-community/Qwen3.5-0.8B-4bit}"
step "4/4: Model smoke test (${MODEL})"

# Cache model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${MODEL}', allow_patterns=['*.json','*.safetensors'])
print('Model cached')
" 2>&1

# Start server
PORT=18999
"${BUILD_DIR}/server" --port ${PORT} > /tmp/ci_server.log 2>&1 &
SERVER_PID=$!

stop_server() {
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
        kill -0 "${SERVER_PID}" 2>/dev/null || return 0
        sleep 1
    done
    kill -9 "${SERVER_PID}" 2>/dev/null || true
}
trap stop_server EXIT

for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "000")
    if [[ "$code" == "200" ]]; then green "  Server ready after ${i}s"; break; fi
    sleep 1
done

# Warm-up
curl -s --max-time 120 -o /dev/null -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"temperature\":0.0,\"stream\":false}" 2>/dev/null || true

# Real test
response=$(curl -s --max-time 180 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"system\",\"content\":\"Answer with just the number.\"},{\"role\":\"user\",\"content\":\"2+2=\"}],\"max_tokens\":16,\"temperature\":0.0,\"stream\":false}" 2>&1)

answer=$(echo "$response" | python3 -c "
import sys,json,re
d=json.load(sys.stdin)
c=d['choices'][0]['message']['content'].strip()
c=re.sub(r'<think>.*?</think>','',c,flags=re.DOTALL)
m=re.search(r'(\d+)',c)
print(m.group(1) if m else '')
" 2>/dev/null) || answer=""

if [[ "$answer" == "4" ]]; then    green "  ✅ Model answered correctly: 2+2=${answer}"; 
else                               red "  ❌ Expected '4', got '${answer}'"; PASS=false; fi

stop_server
