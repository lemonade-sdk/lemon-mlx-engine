#!/bin/bash
#
# Resolve the ROCm tarball URL for a given platform and version.
#
# Uses AMD's official repo tarball distribution:
#   https://repo.amd.com/rocm/tarball/therock-dist-{platform}-{gfx_target}-{version}.tar.gz
#
# Mirrors the pattern used by lemonade-sdk/whisper.cpp-rocm
# (ci/resolve-rocm-version.sh), which is the reference implementation
# requested by @geramyl for the Lemonade stack.
#
# Usage:
#   source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version>
#
# Arguments:
#   platform      - "linux" or "windows"
#   gfx_target    - GPU target (defaults to gfx1151 if not specified or is a group target)
#   rocm_version  - Specific version (e.g. 7.12.0, 7.9.0rc1) - required, no "latest" auto-detection
#
# Outputs (exported):
#   ROCM_RESOLVED_VERSION - The resolved version string
#   ROCM_TARBALL_URL      - The full URL to download
#
# A preflight HEAD probe is performed before exporting ROCM_TARBALL_URL so that
# non-existent (target, version) combos (e.g. 7.2.1 — never published) fail
# here with a clear diagnostic instead of crashing `tar` on an HTML 4xx body.
# Opt out with ROCM_SKIP_URL_PROBE=1 for offline / restricted-egress CI.

platform="$1"
gfx_target="$2"
rocm_version="$3"

if [ -z "$platform" ] || [ -z "$gfx_target" ] || [ -z "$rocm_version" ]; then
    echo "Usage: source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version>"
    return 1 2>/dev/null || exit 1
fi

# Validate that a specific version was provided (no "latest" auto-detection)
if [ "$rocm_version" = "latest" ]; then
    echo "ERROR: 'latest' auto-detection is not supported."
    echo "Please specify a concrete ROCm version (e.g., 7.12.0, 7.9.0rc1)."
    echo "Available versions: https://repo.amd.com/rocm/tarball/"
    return 1 2>/dev/null || exit 1
fi

# Validate version format (should be X.Y.Z or X.Y.ZaNNNNNNNN pattern)
if ! echo "$rocm_version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+'; then
    echo "ERROR: Invalid ROCm version format: '$rocm_version'"
    echo "Expected format: X.Y.Z (e.g., 7.12.0) or X.Y.ZaNNNNNNNN (e.g., 7.11.0a20251205)"
    return 1 2>/dev/null || exit 1
fi

# For the AMD tarball distribution, use gfx1151 as the base target
# The tarball contains ROCm tools/libraries for all supported GPUs
# GPU targets are specified during build via GPU_TARGETS / CMAKE_HIP_ARCHITECTURES
# Group targets (gfx110X, gfx120X, gfx115X_gfx120X) fall back to gfx1151 as the base
base_target="gfx1151"
case "$gfx_target" in
    gfx110X|gfx120X|gfx115X_gfx120X|gfx1150|gfx1100)
        base_target="gfx1151"
        ;;
    *)
        # Use the specific target if it's an individual target
        base_target="$gfx_target"
        ;;
esac

# Construct the AMD official repo URL
ROCM_TARBALL_URL="https://repo.amd.com/rocm/tarball/therock-dist-${platform}-${base_target}-${rocm_version}.tar.gz"

# Preflight HEAD probe. If AMD hasn't published this (platform, target, version)
# combo, fail loudly here rather than letting `tar` crash on a piped HTML 4xx
# body later. Opt out with ROCM_SKIP_URL_PROBE=1 if needed (e.g. offline CI).
if [ "${ROCM_SKIP_URL_PROBE:-0}" != "1" ]; then
    probe_status=$(curl --silent --show-error --location --head \
        --output /dev/null --write-out '%{http_code}' \
        --max-time 30 "$ROCM_TARBALL_URL" || echo "000")
    if [ "$probe_status" != "200" ]; then
        echo "ERROR: ROCm tarball URL is not reachable (HTTP $probe_status)."
        echo "       URL: $ROCM_TARBALL_URL"
        echo "       Either the version '$rocm_version' isn't published for"
        echo "       base target '$base_target' on platform '$platform', or"
        echo "       AMD's repo is temporarily unavailable."
        echo "       Browse https://repo.amd.com/rocm/tarball/ for the current"
        echo "       list of published (platform, target, version) combos."
        return 1 2>/dev/null || exit 1
    fi
fi

export ROCM_RESOLVED_VERSION="$rocm_version"
export ROCM_TARBALL_URL
echo "ROCm version: $ROCM_RESOLVED_VERSION"
echo "ROCm URL: $ROCM_TARBALL_URL"
