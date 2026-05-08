#!/usr/bin/env bash
# copy-litert-gpu-dylibs.sh
#
# Embeds LiteRT-LM GPU acceleration dylibs into the consuming app's bundle
# during the Xcode build. Intended to be invoked from a Run Script Build
# Phase on the consumer's main target.
#
# Why this is needed:
#   LiteRT-LM dlopen()s GPU accelerator dylibs by plain name at runtime
#   (`libLiteRtMetalAccelerator.dylib`, `libLiteRt.dylib`,
#    `libGemmaModelConstraintProvider.dylib`). They must live as plain
#   `@rpath/<name>.dylib` files inside `App.app/Frameworks/`. CocoaPods
#   doesn't natively embed multi-arch plain dylibs from a Pod, so we ship
#   them under `ios/dylibs/{ios_arm64,ios_sim_arm64}/` and copy the right
#   slice here.
#
# Without this script: dlopen fails -> `[gpu_registry] GPU accelerator could
# not be loaded` -> sendMessage runs on CPU (~10-25x slower than GPU).
#
# Inputs (all read from Xcode-provided env vars):
#   PLATFORM_NAME          -> "iphonesimulator" or "iphoneos"
#   ARCHS                  -> e.g. "arm64"
#   TARGET_BUILD_DIR       -> .../Build/Products/Debug-iphonesimulator
#   UNLOCALIZED_RESOURCES_FOLDER_PATH -> e.g. RnMcpDemo.app
#   PODS_ROOT              -> .../ios/Pods
#   EXPANDED_CODE_SIGN_IDENTITY -> for codesign (may be empty on simulator)
set -euo pipefail

if [[ -z "${PLATFORM_NAME:-}" ]]; then
  echo "error: PLATFORM_NAME not set; this script must run inside an Xcode build" >&2
  exit 1
fi

# Locate dylib source. Two possible layouts:
#   1. Inside Pods (when consumed via CocoaPods + vendored): never works
#      because preserve_paths doesn't deploy plain dylibs.
#   2. Inside node_modules (npm consumer)
#   3. Relative path (when Pod is consumed via path:)
DYLIB_DIR=""
if [[ -n "${PODS_ROOT:-}" ]]; then
  # ${PODS_ROOT} is <app>/ios/Pods. node_modules sits at <app>/node_modules,
  # so two levels up. Some monorepos place node_modules at <app>/../node_modules
  # (workspace hoist) — try both.
  for cand in \
    "${PODS_ROOT}/../../node_modules/@zkproofport/rn-litert-gemma4/ios/dylibs" \
    "${PODS_ROOT}/../../node_modules/react-native-litert-lm/ios/dylibs" \
    "${PODS_ROOT}/../../../node_modules/@zkproofport/rn-litert-gemma4/ios/dylibs" \
    "${PODS_ROOT}/../../../node_modules/react-native-litert-lm/ios/dylibs"; do
    if [[ -d "$cand" ]]; then DYLIB_DIR="$cand"; break; fi
  done
fi
if [[ -z "$DYLIB_DIR" && -n "${PODS_TARGET_SRCROOT:-}" ]]; then
  if [[ -d "${PODS_TARGET_SRCROOT}/ios/dylibs" ]]; then
    DYLIB_DIR="${PODS_TARGET_SRCROOT}/ios/dylibs"
  fi
fi
if [[ -z "$DYLIB_DIR" ]]; then
  echo "warning: could not locate ios/dylibs directory; skipping GPU dylib embed" >&2
  echo "         (LiteRT-LM will fall back to CPU at runtime)" >&2
  exit 0
fi

# Pick slice based on platform.
case "$PLATFORM_NAME" in
  iphonesimulator) SLICE="ios_sim_arm64" ;;
  iphoneos)        SLICE="ios_arm64" ;;
  *)
    echo "warning: unknown PLATFORM_NAME=$PLATFORM_NAME; skipping" >&2
    exit 0
    ;;
esac

SRC="$DYLIB_DIR/$SLICE"
if [[ ! -d "$SRC" ]]; then
  echo "warning: slice $SRC missing; skipping GPU dylib embed" >&2
  exit 0
fi

DEST="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/Frameworks"
mkdir -p "$DEST"

DYLIBS=(
  "libLiteRt.dylib"
  "libLiteRtMetalAccelerator.dylib"
  "libGemmaModelConstraintProvider.dylib"
)

for d in "${DYLIBS[@]}"; do
  if [[ ! -f "$SRC/$d" ]]; then
    echo "warning: $SRC/$d missing — skipping" >&2
    continue
  fi
  cp -f "$SRC/$d" "$DEST/$d"
  # Codesign with the project's expanded identity. Empty identity means
  # ad-hoc signing on the simulator, which `-` accepts.
  IDENTITY="${EXPANDED_CODE_SIGN_IDENTITY:--}"
  codesign --force --sign "$IDENTITY" --timestamp=none "$DEST/$d" 2>&1 \
    || codesign --force --sign - --timestamp=none "$DEST/$d"
  echo "embedded: $DEST/$d ($(file -b "$DEST/$d" | head -c 80))"
done
