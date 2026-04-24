#!/bin/bash
#
# Resolve the ROCm tarball URL for a given platform and version spec.
#
# Uses AMD's official repo tarball distribution:
#   https://repo.amd.com/rocm/tarball/therock-dist-{platform}-{gfx_target}-{version}.tar.gz
#
# Mirrors the pattern used by lemonade-sdk/whisper.cpp-rocm
# (ci/resolve-rocm-version.sh), extended with version-prefix auto-discovery
# so the workflow can track the latest 7.12.x patch without hardcoding it.
#
# Usage:
#   source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version_spec>
#
# Arguments:
#   platform            - "linux" or "windows"
#   gfx_target          - GPU target (group targets fall back to gfx1151 base)
#   rocm_version_spec   - Either:
#                           (a) an exact version X.Y.Z[suffix] (e.g. 7.12.0,
#                               7.9.0rc1, 7.11.0a20251205), OR
#                           (b) a version prefix X.Y (e.g. 7.12), which is
#                               resolved against the tarball directory index
#                               to the newest published patch for (platform,
#                               base_target). Pre-release suffixes (rcN, aN,
#                               bN, pre, nightly) are skipped unless the spec
#                               itself is a full rc-tagged version.
#
# Outputs (exported):
#   ROCM_RESOLVED_VERSION - The resolved version string
#   ROCM_TARBALL_URL      - The full URL to download
#
# A preflight HEAD probe is performed before exporting ROCM_TARBALL_URL so that
# non-existent (target, version) combos (e.g. 7.2.1 — never published as a
# tarball) fail here with a clear diagnostic instead of crashing `tar` on an
# HTML 4xx body. Opt out with ROCM_SKIP_URL_PROBE=1 for offline / restricted-
# egress CI.

platform="$1"
gfx_target="$2"
rocm_version_spec="$3"

if [ -z "$platform" ] || [ -z "$gfx_target" ] || [ -z "$rocm_version_spec" ]; then
    echo "Usage: source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version_spec>"
    return 1 2>/dev/null || exit 1
fi

if [ "$rocm_version_spec" = "latest" ]; then
    echo "ERROR: 'latest' is not a supported spec — it hides which ROCm the build"
    echo "       actually used. Pass a prefix like '7.12' (resolved to newest"
    echo "       patch) or an exact version like '7.12.0' instead."
    echo "       Available versions: https://repo.amd.com/rocm/tarball/"
    return 1 2>/dev/null || exit 1
fi

# Group targets all draw from the gfx1151 base tarball — it ships device libs
# for every supported RDNA3/3.5/4 arch, and CMAKE_HIP_ARCHITECTURES narrows it
# at build time. Individual targets pass through as their own base.
base_target="gfx1151"
case "$gfx_target" in
    gfx110X|gfx120X|gfx115X_gfx120X|gfx1150|gfx1100)
        base_target="gfx1151"
        ;;
    *)
        base_target="$gfx_target"
        ;;
esac

tarball_index_url="https://repo.amd.com/rocm/tarball/"

# Decide exact vs prefix mode. Prefix mode = only "major.minor" with no patch.
# Anything with a third dotted component (digit or suffix) is treated as exact.
if echo "$rocm_version_spec" | grep -qE '^[0-9]+\.[0-9]+$'; then
    spec_mode="prefix"
elif echo "$rocm_version_spec" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+'; then
    spec_mode="exact"
else
    echo "ERROR: Invalid ROCm version spec: '$rocm_version_spec'"
    echo "       Expected either a prefix (e.g. 7.12) or an exact version"
    echo "       (e.g. 7.12.0, 7.9.0rc1, 7.11.0a20251205)."
    return 1 2>/dev/null || exit 1
fi

if [ "$spec_mode" = "prefix" ]; then
    echo "Resolving latest ${rocm_version_spec}.x tarball for ${platform}/${base_target}"
    echo "Index: $tarball_index_url"
    listing=$(curl --silent --show-error --location --max-time 30 "$tarball_index_url" || true)
    if [ -z "$listing" ]; then
        echo "ERROR: Could not fetch tarball directory listing from $tarball_index_url"
        return 1 2>/dev/null || exit 1
    fi

    # Escape dots in prefix for regex safety.
    prefix_re=$(echo "$rocm_version_spec" | sed 's/\./\\./g')
    # Pull out every patch version matching {prefix}.N (stable patches only —
    # pre-release suffixes like rcN, aNNN, pre are intentionally excluded here;
    # pin an exact version if you need a pre-release).
    resolved_patch=$(echo "$listing" \
        | grep -oE "therock-dist-${platform}-${base_target}-${prefix_re}\.[0-9]+\.tar\.gz" \
        | sed -E "s/^therock-dist-${platform}-${base_target}-(${prefix_re}\.[0-9]+)\.tar\.gz$/\1/" \
        | sort -uV | tail -1)

    if [ -z "$resolved_patch" ]; then
        echo "ERROR: No ${rocm_version_spec}.x stable tarball published for"
        echo "       ${platform}/${base_target} at $tarball_index_url."
        echo "       Either AMD hasn't shipped a ${rocm_version_spec}.x release for"
        echo "       that (platform, target) yet, or you need to pin an exact"
        echo "       pre-release version (e.g. ${rocm_version_spec}.0rc1) instead"
        echo "       of the '${rocm_version_spec}' prefix."
        return 1 2>/dev/null || exit 1
    fi

    ROCM_RESOLVED_VERSION="$resolved_patch"
else
    ROCM_RESOLVED_VERSION="$rocm_version_spec"
fi

ROCM_TARBALL_URL="https://repo.amd.com/rocm/tarball/therock-dist-${platform}-${base_target}-${ROCM_RESOLVED_VERSION}.tar.gz"

# Preflight HEAD probe. For prefix mode this is a belt-and-suspenders check
# (we already saw the file in the directory listing); for exact mode it's the
# only line of defense against typos in rocm-channels.json.
if [ "${ROCM_SKIP_URL_PROBE:-0}" != "1" ]; then
    probe_status=$(curl --silent --show-error --location --head \
        --output /dev/null --write-out '%{http_code}' \
        --max-time 30 "$ROCM_TARBALL_URL" || echo "000")
    if [ "$probe_status" != "200" ]; then
        echo "ERROR: ROCm tarball URL is not reachable (HTTP $probe_status)."
        echo "       URL: $ROCM_TARBALL_URL"
        echo "       Either the version '$ROCM_RESOLVED_VERSION' isn't published for"
        echo "       base target '$base_target' on platform '$platform', or"
        echo "       AMD's repo is temporarily unavailable."
        echo "       Browse https://repo.amd.com/rocm/tarball/ for the current"
        echo "       list of published (platform, target, version) combos."
        return 1 2>/dev/null || exit 1
    fi
fi

export ROCM_RESOLVED_VERSION
export ROCM_TARBALL_URL
echo "ROCm version: $ROCM_RESOLVED_VERSION"
echo "ROCm URL: $ROCM_TARBALL_URL"
