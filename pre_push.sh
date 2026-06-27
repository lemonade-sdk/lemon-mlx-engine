#!/usr/bin/env bash
# Pre-push: PR review → local CI → push (respect maintainer's credits).
# Usage:
#   bash pre_push.sh [base_ref]   # review + CI + push
#   bash pre_push.sh --push-only  # skip review/CI, just push
set -euo pipefail

START_TS=$(date +%s)
PASS=true

green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
step()   { printf "\033[34m━━━ %s ━━━\033[0m\n" "$*"; }

BASE="${1:-HEAD~1}"
PROJECT="$(git rev-parse --show-toplevel 2>/dev/null || exit 1)"
cd "$PROJECT"

if [ "${1:-}" = "--push-only" ]; then
    step "Push only"
    git push fork main --force
    green "✅ Pushed"
    exit 0
fi

# ── Step 1: PR Review ──
step "1/3: PR Review against ${BASE}"
PR_REVIEW="${HOME}/tools/pr_review.sh"
if [ -x "$PR_REVIEW" ]; then
    bash "$PR_REVIEW" "$BASE" || { PASS=false; }
else
    yellow "  pr_review.sh not found, skipping"
fi

# ── Step 2: Local CI (build + unit tests) ──
step "2/3: Local CI (build + unit tests)"
CI_SCRIPT="${PROJECT}/ci_local.sh"
if [ -x "$CI_SCRIPT" ]; then
    bash "$CI_SCRIPT" --build-only || { PASS=false; }
else
    step "  Building directly..."
    BUILD_DIR="${PROJECT}/build"
    rm -rf "$BUILD_DIR" 2>/dev/null || true
    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    cmake "$PROJECT" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMLX_LM_BUILD_TESTS=ON -DMLX_BUILD_ROCM=OFF \
        -DMINJA_EXAMPLE_ENABLED=OFF 2>&1 | tail -3
    cmake --build . -j "$(nproc)" 2>&1 | tail -3
    ctest --test-dir tests --output-on-failure --timeout 120 2>&1 | tail -3
fi

# ── Step 3: Push (only if all passed) ──
step "3/3: Push to fork"
if $PASS; then
    green "  All checks passed, pushing..."
    git push fork main --force
    green "✅ Pushed successfully"
else
    red "❌ Checks failed — aborting push"
    exit 1
fi

DURATION=$(( $(date +%s) - START_TS ))
green "✅ Done in ${DURATION}s — maintainer credits saved!"
