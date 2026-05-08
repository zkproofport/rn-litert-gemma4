#!/usr/bin/env bash
# build-ios-xcframework.sh
#
# Builds LiteRTLM.xcframework from LiteRT-LM main HEAD source so that the
# wrapper static archive is ABI-compatible with main-HEAD prebuilt GPU dylibs
# (libLiteRt, libLiteRtMetalAccelerator, libLiteRtTopKMetalSampler,
# libGemmaModelConstraintProvider).
#
# The upstream wrapper (react-native-litert-lm@v0.3.6) ships a static archive
# built against the v0.10.2 tag — but the GPU prebuilt dylibs only exist on
# main HEAD (post-v0.10.2). Mixing the two crashes at
# `LiteRtEnvironmentOptionsT::GetOption+88` (SIGSEGV) because that C++ class
# was renamed to `LiteRtEnvironmentT` between v0.10.2 and main.
#
# This script rebuilds ONLY the wrapper static archive at main HEAD and packs
# the prebuilt GPU dylibs alongside, producing an ABI-consistent xcframework.
#
# Prerequisites:
#   - bazelisk (brew install bazelisk)
#   - rustup with aarch64-apple-ios + aarch64-apple-ios-sim targets
#     (rustup target add aarch64-apple-ios aarch64-apple-ios-sim)
#   - Xcode command line tools
#
# Usage:
#   ./scripts/build-ios-xcframework.sh
#
# Output:
#   ios/Frameworks/LiteRTLM.xcframework/                (rebuilt static archive)
#   ios/dylibs/{ios_arm64,ios_sim_arm64}/lib*.dylib     (prebuilt GPU dylibs)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LITERT_SRC="${LITERT_SRC:-/tmp/LiteRT-LM}"
STAGE_DIR="$REPO_ROOT/.litert-lm-build/staged"
FRAMEWORK_NAME="LiteRTLM"
OUTPUT_XCFW="$REPO_ROOT/ios/Frameworks/$FRAMEWORK_NAME.xcframework"

# ---- 0. Verify source clone -------------------------------------------------
if [ ! -d "$LITERT_SRC/.git" ]; then
  echo "Error: $LITERT_SRC missing. Clone first:"
  echo "  git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git $LITERT_SRC"
  exit 1
fi
cd "$LITERT_SRC"
HEAD_SHA=$(git rev-parse HEAD)
HEAD_DATE=$(git log -1 --format='%cI')
echo "==> LiteRT-LM source: $LITERT_SRC"
echo "    HEAD: $HEAD_SHA ($HEAD_DATE)"

# ---- 1. Sanity-check prebuilt dylibs ----------------------------------------
for ARCH in ios_arm64 ios_sim_arm64; do
  for LIB in libLiteRt.dylib libLiteRtMetalAccelerator.dylib libGemmaModelConstraintProvider.dylib; do
    PATH_DYLIB="$LITERT_SRC/prebuilt/$ARCH/$LIB"
    if [ ! -f "$PATH_DYLIB" ]; then
      echo "Error: missing prebuilt $PATH_DYLIB"
      echo "       Make sure git lfs pulled the files (git lfs install && git lfs pull)."
      exit 1
    fi
  done
done
# TopKMetalSampler is device-only on main HEAD
if [ ! -f "$LITERT_SRC/prebuilt/ios_arm64/libLiteRtTopKMetalSampler.dylib" ]; then
  echo "Error: missing libLiteRtTopKMetalSampler.dylib (device)"
  exit 1
fi
echo "    prebuilt dylibs OK"

# ---- 2. Bazel toolchain -----------------------------------------------------
if command -v bazelisk &>/dev/null; then BAZEL="bazelisk"
elif command -v bazel &>/dev/null; then BAZEL="bazel"
else echo "Error: install bazelisk: brew install bazelisk"; exit 1; fi
echo "==> Using Bazel: $($BAZEL --version 2>&1 | head -1)"

# ---- 3. Targets to build ----------------------------------------------------
# Path A: Build ONLY //c:engine (the C API entry-point). Google's official
# ci-build-mac.yml uses //runtime/engine:litert_lm_main as a single end-to-end
# binary with --define=litert_link_capi_so=true (transitive deps go to dylib
# at link-time, not into archive). Wrapper code (HybridLiteRTLM.cpp) calls
# only C API (litert_lm_engine_*, litert_lm_session_*, litert_lm_conversation_*),
# so we don't need to drag vision/embedding/session/proto into the archive.
# Vision/audio path goes through litert_lm_conversation_send_message() with
# JSON-embedded image/audio path — handled inside the dylib.
TARGETS=(
  //c:engine
)

# ---- 4. Build per-config + collect .o files into static archive -------------
mkdir -p "$STAGE_DIR"
BAZEL_OUTPUT_BASE=$($BAZEL info output_base 2>/dev/null)
BAZEL_EXECROOT="$BAZEL_OUTPUT_BASE/execroot"

build_static_for_config() {
  local CONFIG="$1"        # ios_arm64 | ios_sim_arm64
  local CONFIG_DIR="$2"    # ios_arm64-opt | ios_sim_arm64-opt
  local SDK_NAME="$3"
  local TARGET_TRIPLE="$4"
  local OUTPUT_PATH="$5"

  echo ""
  echo "==> Building $CONFIG with prebuilt-matching flags..."
  # Filter mac-only targets when building for iOS Simulator (matches
  # Google's ci-build-mac.yml: --build_tag_filters=...). Also build
  # //runtime/engine:litert_lm_main so Bazel populates ALL transitive deps
  # (Rust crates, sentencepiece, miniaudio, tokenizers_cpp, etc.) into
  # bazel-out — we don't use the binary, just need the .o/.rlib outputs.
  local TAG_FILTERS=""
  if [ "$CONFIG" = "ios_sim_arm64" ]; then
    TAG_FILTERS="--build_tag_filters=-requires-mac-inputs:hard,-no_mac"
  fi

  $BAZEL build "${TARGETS[@]}" //runtime/engine:litert_lm_main \
    --config="$CONFIG" \
    --compilation_mode=opt \
    --copt=-fembed-bitcode-marker \
    --repo_env=USE_PYWRAP_RULES=True \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false \
    $TAG_FILTERS \
    --show_timestamps \
    2>&1 | tail -10

  # Collect .o + .rlib (NOT .a): Bazel's .a archives contain the same .o
  # files we already collected as standalones, so adding .a creates
  # duplicates (e.g. miniaudio.o appearing twice → 1171 dup symbols).
  echo "   collecting .o + .rlib from $CONFIG_DIR/..."
  local CFG_BIN="$BAZEL_EXECROOT/litert_lm/bazel-out/${CONFIG_DIR}/bin"
  local RAW_LIST="$STAGE_DIR/${CONFIG}-raw.txt"
  (find "$CFG_BIN" -name "*.o" ! -name "*.h.processed" 2>/dev/null;
   find "$BAZEL_EXECROOT/litert_lm" -path "*${CONFIG_DIR}*" -name "*.rlib" 2>/dev/null) > "$RAW_LIST"

  # md5 content-hash dedupe (different .o paths sometimes contain
  # bit-identical content; we keep the first occurrence per hash).
  local DEDUPE_LIST="$STAGE_DIR/${CONFIG}-dedupe.txt"
  python3 -c '
import sys, hashlib, os
seen = set()
for line in sys.stdin:
    f = line.strip()
    if not f or not os.path.exists(f):
        continue
    try:
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
    except Exception:
        continue
    if h not in seen:
        seen.add(h)
        print(f)
' < "$RAW_LIST" > "$DEDUPE_LIST"
  local CNT=$(wc -l < "$DEDUPE_LIST" | tr -d ' ')
  echo "   merging $CNT unique objects (deduped from $(wc -l < $RAW_LIST | tr -d ' '))"
  [ "$CNT" -gt 0 ] || { echo "Error: No objects for $CONFIG"; exit 1; }

  xcrun libtool -static -o "$OUTPUT_PATH" -filelist "$DEDUPE_LIST" 2>&1 | grep -v "has no symbols" || true
  echo "   archive size: $(du -h "$OUTPUT_PATH" | cut -f1)"
}

# Extract engine_impl.o into a small LiteRTLMInit.a to be force_load'ed
# by the consumer Podfile. Without this, dead-code stripping removes the
# LITERT_LM_REGISTER_ENGINE static initializers and engine_create fails
# with "Engine type not found: 1".
extract_init_archive() {
  local FULL_ARCHIVE="$1"
  local INIT_OUTPUT="$2"
  local TMP=$(mktemp -d)
  ( cd "$TMP" && ar x "$FULL_ARCHIVE" engine_impl.o && ar rcs LiteRTLMInit.a engine_impl.o )
  cp "$TMP/LiteRTLMInit.a" "$INIT_OUTPUT"
  rm -rf "$TMP"
}

build_static_for_config "ios_arm64" "ios_arm64-opt" \
  "iphoneos" "arm64-apple-ios15.0" \
  "$STAGE_DIR/libengine-device.a"

build_static_for_config "ios_sim_arm64" "ios_sim_arm64-opt" \
  "iphonesimulator" "arm64-apple-ios15.0-simulator" \
  "$STAGE_DIR/libengine-sim.a"

# ---- 5. Build XCFramework ---------------------------------------------------
echo ""
echo "==> Assembling $FRAMEWORK_NAME.xcframework..."
TMP_FW=$(mktemp -d)
trap "rm -rf $TMP_FW" EXIT
C_API_HEADER="$LITERT_SRC/c/engine.h"

build_framework_slice() {
  local ARCH_NAME="$1"     # device | simulator
  local LIB_PATH="$2"
  local PLAT="$3"          # iPhoneOS | iPhoneSimulator
  local FW_DIR="$TMP_FW/$ARCH_NAME/$FRAMEWORK_NAME.framework"
  mkdir -p "$FW_DIR/Headers"
  cp "$LIB_PATH" "$FW_DIR/$FRAMEWORK_NAME"
  cp "$C_API_HEADER" "$FW_DIR/Headers/"
  # Inject LiteRTLMInit.a (engine_impl.o force_load target).
  extract_init_archive "$LIB_PATH" "$FW_DIR/LiteRTLMInit.a"
  cat > "$FW_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>$FRAMEWORK_NAME</string>
  <key>CFBundleIdentifier</key><string>com.google.ai.edge.litert-lm</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>main-${HEAD_SHA:0:7}</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>MinimumOSVersion</key><string>15.0</string>
  <key>CFBundleSupportedPlatforms</key><array><string>$PLAT</string></array>
</dict></plist>
PLIST
}

build_framework_slice "device"    "$STAGE_DIR/libengine-device.a" "iPhoneOS"
build_framework_slice "simulator" "$STAGE_DIR/libengine-sim.a"    "iPhoneSimulator"

rm -rf "$OUTPUT_XCFW"
xcodebuild -create-xcframework \
  -framework "$TMP_FW/device/$FRAMEWORK_NAME.framework" \
  -framework "$TMP_FW/simulator/$FRAMEWORK_NAME.framework" \
  -output "$OUTPUT_XCFW" 2>&1

# ---- 6. Stage prebuilt GPU dylibs ------------------------------------------
echo ""
echo "==> Staging prebuilt GPU dylibs (from main HEAD)..."
for ARCH in ios_arm64 ios_sim_arm64; do
  DEST="$REPO_ROOT/ios/dylibs/$ARCH"
  mkdir -p "$DEST"
  for LIB in libLiteRt.dylib libLiteRtMetalAccelerator.dylib libLiteRtTopKMetalSampler.dylib libGemmaModelConstraintProvider.dylib; do
    SRC="$LITERT_SRC/prebuilt/$ARCH/$LIB"
    if [ -f "$SRC" ]; then
      cp "$SRC" "$DEST/$LIB"
      echo "    $ARCH/$LIB"
    fi
  done
done

echo ""
echo "==> Done."
echo "    XCFramework: $OUTPUT_XCFW"
echo "    Plain dylibs (for App.app/Frameworks/): $REPO_ROOT/ios/dylibs/"
echo "    HEAD: $HEAD_SHA ($HEAD_DATE)"
