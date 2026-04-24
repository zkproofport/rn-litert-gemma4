#!/bin/bash
# build-ios-engine.sh
#
# Builds the LiteRT-LM C engine as a static library for iOS (device + simulator)
# using Bazel, then packages it into an XCFramework for CocoaPods.
#
# Prerequisites:
#   - Bazel 7.6.1+ (via Bazelisk recommended)
#   - Xcode command line tools
#
# Usage:
#   ./scripts/build-ios-engine.sh
#
# Output:
#   ios/Frameworks/LiteRTLM.xcframework/  (static library + headers)

set -euo pipefail

LITERT_LM_REPO="https://github.com/google-ai-edge/LiteRT-LM.git"
FRAMEWORK_NAME="LiteRTLM"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LITERT_LM_VERSION="$(node -e "console.log(require('$PROJECT_ROOT/package.json').litertLm.iosGitTag)")"
OUTPUT_DIR="$PROJECT_ROOT/ios/Frameworks"
C_API_HEADER_DIR="$PROJECT_ROOT/cpp/include"
BUILD_DIR="$PROJECT_ROOT/.litert-lm-build"

echo "==> Building LiteRT-LM ${LITERT_LM_VERSION} C engine for iOS..."
echo ""

# ---- 1. Clone / update the LiteRT-LM repo --------------------------------
echo "==> Step 1: Preparing LiteRT-LM source..."
if [ -f "$BUILD_DIR/LiteRT-LM/.bazelrc" ] && [ -f "$BUILD_DIR/LiteRT-LM/requirements.txt" ]; then
  echo "   Source already exists, checking out ${LITERT_LM_VERSION}..."
  cd "$BUILD_DIR/LiteRT-LM"
  git fetch --tags 2>/dev/null || true
  git checkout "$LITERT_LM_VERSION" 2>/dev/null || git checkout "tags/$LITERT_LM_VERSION"
else
  # Clean up any failed previous clone
  rm -rf "$BUILD_DIR/LiteRT-LM"
  mkdir -p "$BUILD_DIR"
  echo "   Cloning LiteRT-LM (shallow, skipping LFS)..."

  # Clone without LFS filter to avoid requiring git-lfs installation.
  # The prebuilt/ directory will contain LFS pointer files but we don't
  # need those — we're building the engine from source.
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch "$LITERT_LM_VERSION" \
    -c filter.lfs.smudge=cat \
    -c filter.lfs.process= \
    -c filter.lfs.clean=cat \
    -c filter.lfs.required=false \
    "$LITERT_LM_REPO" "$BUILD_DIR/LiteRT-LM"

  cd "$BUILD_DIR/LiteRT-LM"
fi

LITERT_SRC="$BUILD_DIR/LiteRT-LM"

# ---- 1b. Apply iOS-specific patches ---------------------------------------
# These patches fix:
# - mmap PROT_WRITE removal (iOS rejects CoW for large files)
# - Error capture API (litert_lm_get_last_error)
# - Engine registerer moved outside anonymous namespace (iOS linker stripping)
# - Minijinja/Rust stub replacement (custom C++ prompt template)
PATCHES_DIR="$PROJECT_ROOT/scripts/patches"
if [ -d "$PATCHES_DIR" ]; then
  for PATCH_FILE in "$PATCHES_DIR"/*.patch; do
    if [ -f "$PATCH_FILE" ]; then
      echo "   Applying patch: $(basename "$PATCH_FILE")..."
      cd "$LITERT_SRC"
      git apply --check "$PATCH_FILE" 2>/dev/null && \
        git apply "$PATCH_FILE" || \
        echo "   (patch already applied or conflicts, skipping)"
    fi
  done
fi

# ---- 2. Verify Bazel is available -----------------------------------------
echo ""
echo "==> Step 2: Checking Bazel..."
if command -v bazelisk &>/dev/null; then
  BAZEL="bazelisk"
elif command -v bazel &>/dev/null; then
  BAZEL="bazel"
else
  echo "Error: Bazel is not installed."
  echo "Install via: brew install bazelisk"
  echo "Or download from: https://github.com/bazelbuild/bazelisk"
  exit 1
fi

BAZEL_VERSION=$($BAZEL --version 2>&1 | head -1)
echo "   Using: $BAZEL ($BAZEL_VERSION)"

# ---- 3. Build the C engine static library for iOS -------------------------
echo ""
echo "==> Step 3: Building //c:engine for iOS..."

cd "$LITERT_SRC"

# Get Bazel output base (where all .o files live in the actual filesystem)
BAZEL_OUTPUT_BASE=$($BAZEL info output_base 2>/dev/null)
BAZEL_EXECROOT="$BAZEL_OUTPUT_BASE/execroot"

STAGE_DIR="$BUILD_DIR/staged-libs"
mkdir -p "$STAGE_DIR"

# Helper: build for a config, then merge ALL transitive .o files into one .a
# Bazel's cc_library produces thin archives — the engine's transitive deps
# (absl, protobuf, runtime, KleidiAI, etc.) are separate .o files that must
# be merged into a single self-contained static library for Xcode.
build_fat_static_lib() {
  local CONFIG="$1"
  local CONFIG_DIR="$2"  # e.g. "ios_arm64-opt" or "ios_sim_arm64-opt"
  local OUTPUT_PATH="$3"

  echo "   Building for $CONFIG..."
  # Build both the engine AND all cc_proto_library targets in a single Bazel
  # invocation. Bazel's cc_proto_library compiles proto-generated code, but
  # the .pb.o files only appear in the output tree if these targets are
  # explicitly requested alongside the engine.
  $BAZEL build \
    //c:engine \
    @sentencepiece//:sentencepiece_cc_proto \
    @sentencepiece//:sentencepiece_model_cc_proto \
    //runtime/proto:engine_cc_proto \
    //runtime/proto:sampler_params_cc_proto \
    //runtime/proto:llm_metadata_cc_proto \
    //runtime/proto:token_cc_proto \
    //runtime/proto:llm_model_type_cc_proto \
    //runtime/util:external_file_cc_proto \
    @com_google_protobuf//:protobuf \
    @com_googlesource_code_re2//:re2 \
    --config=$CONFIG 2>&1 | tail -5

  echo "   Collecting transitive object files from $CONFIG_DIR..."
  local OBJ_LIST="$STAGE_DIR/${CONFIG}-objects.txt"
  find "$BAZEL_EXECROOT" -path "*/${CONFIG_DIR}/bin/*" -name "*.o" \
    ! -name "*.h.processed" 2>/dev/null | sort > "$OBJ_LIST"

  # ---- Compile stubs for Rust/llguidance deps (unavailable on iOS) ----------
  local EXTRA_OBJS="$STAGE_DIR/${CONFIG}-extra-objs"
  rm -rf "$EXTRA_OBJS"
  mkdir -p "$EXTRA_OBJS"

  local STUBS_DIR="$PROJECT_ROOT/scripts/stubs"
  local STUB_FILES=$(find "$STUBS_DIR" \( -name "*.cc" -o -name "*.c" \) 2>/dev/null)
  if [ -n "$STUB_FILES" ]; then
    echo "   Compiling stubs for unavailable dependencies..."
    local SDK_NAME="iphoneos"
    local TARGET_TRIPLE="arm64-apple-ios15.0"
    if [[ "$CONFIG_DIR" == *"sim"* ]]; then
      SDK_NAME="iphonesimulator"
      TARGET_TRIPLE="arm64-apple-ios15.0-simulator"
    fi
    local SDK_PATH=$(xcrun --sdk "$SDK_NAME" --show-sdk-path)
    
    for STUB_SRC in $STUB_FILES; do
      local STUB_BASE=$(basename "$STUB_SRC")
      local STUB_NAME="${STUB_BASE%.*}"
      local STUB_EXT="${STUB_BASE##*.}"
      echo "     → $STUB_NAME ($STUB_EXT)"
      
      if [ "$STUB_EXT" = "cc" ]; then
        xcrun clang++ -c -std=c++20 \
          -target "$TARGET_TRIPLE" \
          -isysroot "$SDK_PATH" \
          -DNDEBUG \
          -o "$EXTRA_OBJS/${STUB_NAME}.o" \
          "$STUB_SRC" 2>&1 || true
      else
        xcrun clang -c \
          -target "$TARGET_TRIPLE" \
          -isysroot "$SDK_PATH" \
          -DNDEBUG \
          -o "$EXTRA_OBJS/${STUB_NAME}.o" \
          "$STUB_SRC" 2>&1 || true
      fi
    done
    
    # Add successfully compiled stubs to the object list
    find "$EXTRA_OBJS" -name "*.o" -size +0c >> "$OBJ_LIST"
  fi

  local OBJ_COUNT=$(wc -l < "$OBJ_LIST" | tr -d ' ')
  echo "   Found $OBJ_COUNT total object files (including proto + stubs)"

  if [ "$OBJ_COUNT" -eq 0 ]; then
    echo "Error: No object files found for $CONFIG in $CONFIG_DIR"
    exit 1
  fi

  # Merge all .o files into a single fat static library using libtool
  echo "   Merging into fat static library..."
  xcrun libtool -static -o "$OUTPUT_PATH" -filelist "$OBJ_LIST" 2>&1 | grep -v "has no symbols" || true

  local LIB_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
  echo "   ✅ $CONFIG: $LIB_SIZE ($OBJ_COUNT objects)"
}

# Build for device (arm64)
DEVICE_LIB="$STAGE_DIR/libengine-device.a"
build_fat_static_lib "ios_arm64" "ios_arm64-opt" "$DEVICE_LIB"

echo ""

# Build for simulator (sim_arm64)
SIM_LIB="$STAGE_DIR/libengine-sim.a"
build_fat_static_lib "ios_sim_arm64" "ios_sim_arm64-opt" "$SIM_LIB"

echo ""
echo "   Device lib: $DEVICE_LIB ($(du -h "$DEVICE_LIB" | cut -f1))"
echo "   Simulator lib: $SIM_LIB ($(du -h "$SIM_LIB" | cut -f1))"

# ---- 4. Copy the C API header ---------------------------------------------
echo ""
echo "==> Step 4: Vendoring C API header..."
mkdir -p "$C_API_HEADER_DIR"
cp "$LITERT_SRC/c/engine.h" "$C_API_HEADER_DIR/litert_lm_engine.h"
echo "   ✅ Copied engine.h → cpp/include/litert_lm_engine.h"

# ---- 5. Create XCFramework from static libraries --------------------------
echo ""
echo "==> Step 5: Creating XCFramework..."

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# Create framework bundles from static libraries
for ARCH_NAME in "device" "simulator"; do
  if [ "$ARCH_NAME" = "device" ]; then
    LIB_PATH="$DEVICE_LIB"
  else
    LIB_PATH="$SIM_LIB"
  fi

  FW_DIR="$TMP_DIR/$ARCH_NAME/$FRAMEWORK_NAME.framework"
  mkdir -p "$FW_DIR/Headers"

  # Copy static lib as the framework binary
  cp "$LIB_PATH" "$FW_DIR/$FRAMEWORK_NAME"

  # Copy headers
  cp "$C_API_HEADER_DIR/litert_lm_engine.h" "$FW_DIR/Headers/"

  # Create Info.plist
  cat > "$FW_DIR/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>${FRAMEWORK_NAME}</string>
  <key>CFBundleIdentifier</key>
  <string>com.google.ai.edge.litert-lm</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${FRAMEWORK_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>0.10.2</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>MinimumOSVersion</key>
  <string>15.0</string>
</dict>
</plist>
PLIST
done

# Create XCFramework
xcodebuild -create-xcframework \
  -framework "$TMP_DIR/device/$FRAMEWORK_NAME.framework" \
  -framework "$TMP_DIR/simulator/$FRAMEWORK_NAME.framework" \
  -output "$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework" 2>&1

echo "   ✅ XCFramework created at: ios/Frameworks/${FRAMEWORK_NAME}.xcframework"

# ---- 6. Create zip for release asset --------------------------------------
echo ""
echo "==> Step 6: Creating release asset zip..."
cd "$OUTPUT_DIR"
zip -r "$PROJECT_ROOT/LiteRTLM-ios-frameworks.zip" . -x ".*" 2>&1
ZIP_SIZE=$(du -h "$PROJECT_ROOT/LiteRTLM-ios-frameworks.zip" | cut -f1)
echo "   ✅ Created LiteRTLM-ios-frameworks.zip (${ZIP_SIZE})"

echo ""
echo "==> Done! iOS engine built and packaged."
echo ""
echo "Contents:"
find "$OUTPUT_DIR" -type f | head -20 | while read f; do
  echo "  $(echo "$f" | sed "s|$PROJECT_ROOT/||") ($(du -h "$f" | cut -f1))"
done
