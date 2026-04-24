#!/bin/bash
#
# Usage: source ci/resolve-rocm-version.sh <platform> <gfx_target> <spec>
#   spec is either a prefix (e.g. 7.12 → newest patch) or an exact
#   X.Y.Z[suffix] (e.g. 7.12.0, 7.9.0rc1).
# Exports ROCM_RESOLVED_VERSION and ROCM_TARBALL_URL.
# Set ROCM_SKIP_URL_PROBE=1 to skip the HEAD preflight.

platform="$1"
gfx_target="$2"
rocm_version_spec="$3"

if [ -z "$platform" ] || [ -z "$gfx_target" ] || [ -z "$rocm_version_spec" ]; then
    echo "Usage: source ci/resolve-rocm-version.sh <platform> <gfx_target> <rocm_version_spec>"
    return 1 2>/dev/null || exit 1
fi

if [ "$rocm_version_spec" = "latest" ]; then
    echo "ERROR: 'latest' is not a supported spec. Pass a prefix like '7.12' or an exact version like '7.12.0'."
    echo "       Available versions: https://repo.amd.com/rocm/tarball/"
    return 1 2>/dev/null || exit 1
fi

# Group targets draw from the gfx1151 base tarball — it ships device libs for
# every supported arch, narrowed at build time via CMAKE_HIP_ARCHITECTURES.
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

if echo "$rocm_version_spec" | grep -qE '^[0-9]+\.[0-9]+$'; then
    spec_mode="prefix"
elif echo "$rocm_version_spec" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+'; then
    spec_mode="exact"
else
    echo "ERROR: Invalid ROCm version spec: '$rocm_version_spec'"
    echo "       Expected either a prefix (e.g. 7.12) or an exact version (e.g. 7.12.0, 7.9.0rc1)."
    return 1 2>/dev/null || exit 1
fi

if [ "$spec_mode" = "prefix" ]; then
    echo "Resolving latest ${rocm_version_spec}.x tarball for ${platform}/${base_target}"
    listing=$(curl --silent --show-error --location --max-time 30 "$tarball_index_url" || true)
    if [ -z "$listing" ]; then
        echo "ERROR: Could not fetch tarball directory listing from $tarball_index_url"
        return 1 2>/dev/null || exit 1
    fi

    prefix_re=$(echo "$rocm_version_spec" | sed 's/\./\\./g')
    # Stable patches only — pre-release suffixes (rcN, aN, pre) are skipped
    # unless the caller pins an exact version.
    resolved_patch=$(echo "$listing" \
        | grep -oE "therock-dist-${platform}-${base_target}-${prefix_re}\.[0-9]+\.tar\.gz" \
        | sed -E "s/^therock-dist-${platform}-${base_target}-(${prefix_re}\.[0-9]+)\.tar\.gz$/\1/" \
        | sort -uV | tail -1)

    if [ -z "$resolved_patch" ]; then
        echo "ERROR: No ${rocm_version_spec}.x stable tarball published for ${platform}/${base_target} at $tarball_index_url."
        echo "       Pin an exact pre-release (e.g. ${rocm_version_spec}.0rc1) if AMD has only shipped RCs."
        return 1 2>/dev/null || exit 1
    fi

    ROCM_RESOLVED_VERSION="$resolved_patch"
else
    ROCM_RESOLVED_VERSION="$rocm_version_spec"
fi

ROCM_TARBALL_URL="https://repo.amd.com/rocm/tarball/therock-dist-${platform}-${base_target}-${ROCM_RESOLVED_VERSION}.tar.gz"

if [ "${ROCM_SKIP_URL_PROBE:-0}" != "1" ]; then
    probe_status=$(curl --silent --show-error --location --head \
        --output /dev/null --write-out '%{http_code}' \
        --max-time 30 "$ROCM_TARBALL_URL" || echo "000")
    if [ "$probe_status" != "200" ]; then
        echo "ERROR: ROCm tarball URL is not reachable (HTTP $probe_status)."
        echo "       URL: $ROCM_TARBALL_URL"
        echo "       Browse https://repo.amd.com/rocm/tarball/ for published (platform, target, version) combos."
        return 1 2>/dev/null || exit 1
    fi
fi

export ROCM_RESOLVED_VERSION
export ROCM_TARBALL_URL
echo "ROCm version: $ROCM_RESOLVED_VERSION"
echo "ROCm URL: $ROCM_TARBALL_URL"
