#!/bin/bash
# download-ios-frameworks.sh
#
# Downloads prebuilt LiteRT-LM iOS static engine from this project's GitHub
# releases. If the prebuilt asset is not available, falls back to building
# from source via Bazel (see build-ios-engine.sh).
#
# The XCFramework contains a static library compiled from the LiteRT-LM
# C engine (//c:engine Bazel target) for both device (arm64) and simulator
# (sim_arm64).
#
# Usage:
#   ./scripts/download-ios-frameworks.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/ios/Frameworks"
C_API_HEADER_DIR="$PROJECT_ROOT/cpp/include"

LITERT_LM_VERSION="$(node -e "console.log(require('$PROJECT_ROOT/package.json').litertLm.iosGitTag)")"
GITHUB_RAW="https://github.com/google-ai-edge/LiteRT-LM/raw/${LITERT_LM_VERSION}"

# Read version from package.json
PACKAGE_VERSION=$(node -e "console.log(require('$PROJECT_ROOT/package.json').version)" 2>/dev/null || echo "0.0.0")
GITHUB_REPO="hung-yueh/react-native-litert-lm"
ASSET_NAME="LiteRTLM-ios-frameworks.zip"

# Skip if already present
if [ -d "$OUTPUT_DIR" ] && [ "$(find "$OUTPUT_DIR" -name "*.xcframework" 2>/dev/null | wc -l)" -gt 0 ]; then
  echo "[LiteRT-LM] iOS frameworks already present at ios/Frameworks/, skipping."
  exit 0
fi

# ---- Ensure C API header is vendored --------------------------------------
echo "[LiteRT-LM] Vendoring C API header..."
mkdir -p "$C_API_HEADER_DIR"
curl -fsSL -o "$C_API_HEADER_DIR/litert_lm_engine.h" \
  "${GITHUB_RAW}/c/engine.h" 2>/dev/null || true

# ---- Try downloading prebuilt from our GitHub releases --------------------
RELEASE_URL="https://github.com/${GITHUB_REPO}/releases/download/v${PACKAGE_VERSION}/${ASSET_NAME}"

echo "[LiteRT-LM] Attempting to download prebuilt iOS engine from:"
echo "   ${RELEASE_URL}"

TMP_ZIP="$PROJECT_ROOT/.ios-frameworks-tmp.zip"
if curl -fsSL -o "$TMP_ZIP" "$RELEASE_URL" 2>/dev/null; then
  echo "[LiteRT-LM] Download successful, extracting..."
  rm -rf "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  unzip -o -q "$TMP_ZIP" -d "$OUTPUT_DIR"
  rm -f "$TMP_ZIP"
  echo "[LiteRT-LM] ✅ iOS frameworks installed from prebuilt release."
  exit 0
fi

rm -f "$TMP_ZIP"
echo "[LiteRT-LM] Prebuilt not available for v${PACKAGE_VERSION}."

# ---- Fall back to building from source ------------------------------------
echo "[LiteRT-LM] Falling back to building from source via Bazel..."
echo ""

if [ -x "$SCRIPT_DIR/build-ios-engine.sh" ]; then
  exec "$SCRIPT_DIR/build-ios-engine.sh"
else
  echo "Error: build-ios-engine.sh not found or not executable."
  echo "Run manually: ./scripts/build-ios-engine.sh"
  exit 1
fi
