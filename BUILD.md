# Build Guide: `@zkproofport/rn-litert-gemma4`

Hard fork of `hung-yueh/react-native-litert-lm@v0.3.6` rebuilt against
LiteRT-LM **main HEAD** (post-v0.10.2) for ABI-compatible integration with
Google's prebuilt iOS GPU dylibs (Metal accelerator + FST constraint
provider). Native Gemma 4 function calling + 22-23 tps inference verified
in iOS Simulator (3/3 autonomous tool-call scenarios pass).

## TL;DR

```bash
# 1. Clone LiteRT-LM main HEAD with LFS
git clone --depth 1 https://github.com/google-ai-edge/LiteRT-LM.git /tmp/LiteRT-LM
cd /tmp/LiteRT-LM && git lfs install && git lfs pull

# 2. Install Rust iOS targets (toolchain auto-registered by Bazel)
rustup target add aarch64-apple-ios aarch64-apple-ios-sim

# 3. Build wrapper static archive (Bazel + libtool)
cd ~/Workspace/rn-litert-gemma4
./scripts/build-ios-xcframework.sh

# 4. Merge augmented archive (Rust crates + alloc shim)
./scripts/merge-augmented-archive.sh    # see "Augmented archive" section

# 5. Use as a regular RN dependency
cd ~/Workspace/RnMcpDemo
npx pod-install  # or: cd ios && pod install
```

## Why a fork?

The upstream `react-native-litert-lm@v0.3.6` ships a static archive built
against LiteRT-LM **v0.10.2**. Google has since updated `prebuilt/ios_*`
GPU dylibs to **main HEAD** (post-v0.10.2). Mixing the two crashes at
`LiteRtEnvironmentOptionsT::GetOption+88` because the C++ class was
renamed to `LiteRtEnvironmentT` between releases.

We rebuild the wrapper static archive at the same `LITERT_REF`
(`47615eb6...`, 2026-04-27) Google used for the prebuilt dylibs, restoring
ABI consistency.

## Why the previous "fundamental wall" conclusion was wrong

A prior diagnostic concluded that mixing the wrapper archive with prebuilt
GPU dylibs caused unfixable heap corruption from duplicate transitive dep
state (absl/RE2/protobuf). That conclusion was **incorrect**:

- `HybridLiteRTLM.cpp` only includes `litert_lm_engine.h` (C API). It does
  not call any `litert::lm::*` C++ symbols.
- prebuilt dylibs are **self-contained** (`absl` 0 exported, 2850
  file-local, 0 undefined). The dylib's absl is naturally isolated.
- Archive's absl uses inline namespace `lts_20260107`; dylib's absl has no
  `lts_*` prefix. The two namespaces don't collide at link time and don't
  share global state at runtime — they're literally different types as far
  as the linker is concerned.

The actual root cause of the earlier breakage was just **wrong build
target selection** (vision/embedding/session pulled in too many
unnecessary `.o` files, and `.a` archives were collected alongside their
own contents → duplicate `miniaudio.o`).

## Architecture

```
+------------------+        +-------------------+
| your RN app      |        | App.app/Frameworks/  ← Run Script copies these
|  HybridLiteRTLM  |        |  libLiteRt.dylib       (dlopen at runtime)
|  (C API only)    |───────▶|  libLiteRtMetalAccelerator.dylib
+------------------+        |  libLiteRtTopKMetalSampler.dylib (device only)
        ▲                   +-------------------+
        │ link-time
        │
+----------------------+
| LiteRTLM.xcframework |   ← `vendored_frameworks` in podspec
|   ios-arm64/         |
|     LiteRTLM         |   ← 242MB augmented archive (LiteRT-LM C++ +
|     LiteRTLMInit.a   |     LiteRT runtime + Rust crates + alloc shim)
|   ios-arm64-simulator/
|     LiteRTLM         |   ← 243MB
|     LiteRTLMInit.a   |
+----------------------+
        ▲
        │ link-time (-Wl,-force_load on LiteRTLMInit.a)
        │
+-------------------------------+
| GemmaModelConstraintProvider  |   ← `vendored_frameworks`
|   .xcframework                |     (FST data processor, link-time)
+-------------------------------+
```

### dylib disposition

| dylib                                 | ios_arm64 | ios_sim_arm64 | how it's used                    |
|---------------------------------------|-----------|---------------|----------------------------------|
| libLiteRt.dylib                       | ✅        | ✅            | dlopen by `litert_runtime`       |
| libLiteRtMetalAccelerator.dylib       | ✅        | ✅            | dlopen by `gpu_registry.cc`      |
| libLiteRtTopKMetalSampler.dylib       | ✅        | ❌ (device only)| dlopen for top-K sampling      |
| libGemmaModelConstraintProvider.dylib | ✅        | ✅            | link-time (xcframework wrap)     |

`gpu_registry.cc` calls `dlopen("libLiteRtMetalAccelerator.dylib")` with
the *plain* library name. dylibs must therefore live at
`App.app/Frameworks/<plain>.dylib` (not inside `*.framework` bundles).

## Build flow

### Step 1: Bazel-build wrapper archives

`scripts/build-ios-xcframework.sh` runs `bazelisk build` for `//c:engine`
under both ios configs:

```bash
bazelisk build //c:engine \
  --config=ios_arm64                  # or ios_sim_arm64
  --compilation_mode=opt
  --copt=-fembed-bitcode-marker
  --repo_env=USE_PYWRAP_RULES=True
  --define=litert_link_capi_so=true
  --define=resolve_symbols_in_exec=false
  --build_tag_filters=-requires-mac-inputs:hard,-no_mac   # sim only
```

These flags match Google's `ci-build-mac.yml` workflow. The
`litert_link_capi_so` define tells Bazel to convert LiteRT (the
TensorFlow Lite runtime under `external/litert/`) into a dynamic-link
target — but LiteRT-LM's own modules (`runtime/...`) still emit `.o` files
that are collected into the wrapper archive.

### Step 2: Build the rest of the dependency graph (`litert_lm_main`)

```bash
bazelisk build //runtime/engine:litert_lm_main \
  --config=ios_sim_arm64 \
  --compilation_mode=opt --copt=-fembed-bitcode-marker \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false
```

This pulls in **everything** the wrapper depends on transitively, including:

- `runtime/components/rust:minijinja_template` (chat-template engine, Rust)
- `runtime/components/tool_use/rust:parsers` (FC parser, Rust)
- `external/crate_index/llguidance` (FST constraint, Rust)
- `external/sentencepiece` (proto + tokenizer)
- `external/miniaudio`
- `external/litert/...` (TFLite runtime)
- `tokenizers_cpp` (HuggingFace BPE)

We don't use the `litert_lm_main` binary itself — we just need Bazel to
populate `bazel-out/{ios_*}-opt/bin/` with the `.o`, `.rlib`, and `.a`
files we'll merge.

### Step 3: Augmented archive (md5-deduped `.o` + `.rlib`)

`xcrun libtool -static` can ingest `.rlib` files directly (it warns
"`lib.rmeta` has no symbols" but the metadata gets ignored). We collect
**only `.o` and `.rlib`** files — never `.a`, because Bazel's `.a`
archives contain the same `.o` files we already collected as standalone
objects (this caused 1171 duplicate symbols of `miniaudio.o` in earlier
attempts).

```bash
EXEC=/private/var/tmp/_bazel_nhn/<HASH>/execroot/litert_lm
SIM_BIN=$EXEC/bazel-out/ios_sim_arm64-opt/bin

(find "$SIM_BIN"  -name "*.o" ! -name "*.h.processed";
 find "$EXEC" -path "*ios_sim_arm64*" -name "*.rlib") | \
  python3 dedupe-by-md5.py > merge-list.txt

xcrun libtool -static -o libengine-sim-full.a -filelist merge-list.txt
```

Result: ~243 MB merged archive containing **everything** the wrapper
needs. Repeat for `ios_arm64-opt` to produce the device archive.

### Step 4: Extract `LiteRTLMInit.a`

```bash
ar x libengine-sim-full.a engine_impl.o
ar rcs LiteRTLMInit.a engine_impl.o
```

This 158 KB sub-archive is `-Wl,-force_load`'ed by the consumer Podfile
to keep the `LITERT_LM_REGISTER_ENGINE` static initializers alive in the
final link (otherwise dead-code stripping nukes the engine factory).

### Step 5: `rust_alloc_shim.c`

Rust `.rlib` files reference `__rust_alloc`, `__rust_dealloc`,
`__rust_realloc`, `__rust_alloc_zeroed`, and `__rust_alloc_error_handler`.
These are normally synthesized by `cargo` or `rustc` at the final link
step, but we're linking the Rust libraries into a non-Rust app — so we
provide a `posix_memalign`-based stub at `cpp/rust_alloc_shim.c`.

The podspec was extended to include `*.c` in `s.source_files` so
Cocoapods compiles the shim along with the existing C++ wrapper code:

```ruby
s.source_files = [
  "cpp/**/*.{hpp,cpp,h,c}",   # +c for rust_alloc_shim.c
  "ios/**/*.{m,mm}",
  "nitrogen/generated/ios/**/*.{mm,swift}",
]
```

## Hermes regex limitation (parser fix) ⚠️

Gemma 4 in autonomous mode (`enableConstrainedDecoding: false`) emits
function calls as a markdown-fenced JSON code block:

````
```json
{
  "tool_name": "echo",
  "parameters": { "text": "hello world" }
}
```
````

A natural regex like `/```(?:json)?\s*\n?(\{[\s\S]*?\})\s*\n?```/` works
correctly under Node.js + JavaScriptCore, **but silently fails to match
under Hermes** (the default React Native runtime since 0.72). Verified by:
- raw input has codepoints `60 60 60` (three backticks) at index 0
- `raw.indexOf('```')` returns 0
- the regex returns `null` anyway

Workaround in `src/index.ts` — `extractJsonFenceBody()` uses
`indexOf('```')` + `lastIndexOf('}')` + `slice` instead of regex:

```typescript
function extractJsonFenceBody(text: string): string | null {
  const start = text.indexOf('```');
  if (start < 0) return null;
  const end = text.indexOf('```', start + 3);
  if (end <= start) return null;
  const inner = text.slice(start + 3, end);
  const braceStart = inner.indexOf('{');
  const braceEnd = inner.lastIndexOf('}');
  if (braceStart < 0 || braceEnd <= braceStart) return null;
  return inner.slice(braceStart, braceEnd + 1);
}
```

`parseGemma4ToolCall()` now recognises both shapes:
- `{"tool_name": "X", "parameters": {...}}` (Gemma chat template)
- `{"name": "X", "arguments": {...}}` (OpenAI style)

**Lesson:** in any RN code that runs under Hermes, prefer
`indexOf`/`slice` over multi-line dotall regex with backtick fences.

## Verified results (sim, iPhone 17 Pro Simulator, 2026-04-30)

`enableConstrainedDecoding: false` (autonomous tool selection):

| scenario  | tps   | TTFT | total | tool detected                              |
|-----------|-------|------|-------|--------------------------------------------|
| echo      | 23.46 | 0.41s| 2.59s | ✅ `echo({"text":"hello world"})`           |
| add       | 22.28 | 0.41s| 2.31s | ✅ `add_numbers({"numbers":[7,13]})`        |
| weather   | 22.22 | 0.41s| 5.19s | ✅ `get_weather({"location":"Seoul"})`      |

3/3 PASS. Compares well to Edge Gallery's 30+ tps device-GPU runs (we're
running on simulator CPU here; device GPU verification still pending).

## Consumer integration (Podfile post_install)

```ruby
post_install do |installer|
  react_native_post_install(installer, config[:reactNativePath],
                            :mac_catalyst_enabled => false)

  # Exclude x86_64 from simulator builds (LiteRTLM has no x86_64 slice).
  installer.pods_project.build_configurations.each do |config|
    config.build_settings['EXCLUDED_ARCHS[sdk=iphonesimulator*]'] = 'x86_64'
  end

  # Force-load LiteRTLMInit.a so static initializers (engine factory) survive.
  installer.aggregate_targets.each do |agg|
    agg.user_project.targets.each do |t|
      next unless t.name == 'YourAppTarget'
      t.build_configurations.each do |c|
        c.build_settings['OTHER_LDFLAGS'] = [
          '$(inherited)',
          '-Wl,-force_load,"${BUILT_PRODUCTS_DIR}/XCFrameworkIntermediates/react-native-litert-lm/LiteRTLM.framework/LiteRTLMInit.a"',
        ].join(' ')
      end
      t.project.save
    end
  end
end
```

You'll also need a Run Script build phase that copies the plain GPU dylibs
into `App.app/Frameworks/` and re-codesigns them. Add this in Xcode under
your app target → Build Phases → "+" → New Run Script Phase, *after* the
"Embed Frameworks" phase:

```bash
# Run Script: Copy LiteRT-LM GPU dylibs
set -e
PKG_ROOT="${PROJECT_DIR}/../node_modules/@zkproofport/rn-litert-gemma4"

# Pick the right slice based on PLATFORM_NAME / EFFECTIVE_PLATFORM_NAME.
case "${PLATFORM_NAME}" in
  iphoneos)             SRC_ARCH="ios_arm64" ;;
  iphonesimulator)      SRC_ARCH="ios_sim_arm64" ;;
  *) echo "Unsupported platform: ${PLATFORM_NAME}" && exit 1 ;;
esac

SRC_DIR="${PKG_ROOT}/ios/dylibs/${SRC_ARCH}"
DST_DIR="${BUILT_PRODUCTS_DIR}/${FRAMEWORKS_FOLDER_PATH}"

mkdir -p "${DST_DIR}"
for lib in libLiteRt.dylib libLiteRtMetalAccelerator.dylib \
           libLiteRtTopKMetalSampler.dylib; do
  if [ -f "${SRC_DIR}/${lib}" ]; then
    echo "  copying ${SRC_ARCH}/${lib}"
    cp -f "${SRC_DIR}/${lib}" "${DST_DIR}/${lib}"
    if [ -n "${EXPANDED_CODE_SIGN_IDENTITY}" ]; then
      codesign --force --sign "${EXPANDED_CODE_SIGN_IDENTITY}" \
               "${DST_DIR}/${lib}"
    fi
  fi
done
```

**Input/Output paths** (so Xcode caches the phase correctly):
- Input file lists: leave empty, or point at `$(SRCROOT)/.../dylibs/$(PLATFORM_NAME)/*.dylib`
- Output file: `$(BUILT_PRODUCTS_DIR)/$(FRAMEWORKS_FOLDER_PATH)/libLiteRt.dylib`

## iOS memory entitlements ⚠️ REQUIRED on device for Gemma 4 E2B+

**Without these entitlements your sendMessage will fail at runtime with `Failed to allocate tensors` on iPhone**, even on an 8GB device. iPhone's default jetsam threshold is ~3GB; Gemma 4 E2B (2.4GB on disk) plus KV cache + activation buffers crosses that limit. This is invisible in the simulator (which has no jetsam cap), so **everything passes on sim and crashes only on device** — easy to misdiagnose as an ABI/wrapper bug.

Backend choice does NOT help: TFLite's CPU path also goes through XNNPack delegate and hits the same allocation cap. You MUST add the entitlements.

### Required entitlements file

Create `ios/<YourApp>/<YourApp>.entitlements`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.developer.kernel.increased-memory-limit</key>
  <true/>
  <key>com.apple.developer.kernel.extended-virtual-addressing</key>
  <true/>
  <key>com.apple.developer.kernel.increased-debugging-memory-limit</key>
  <true/>
</dict>
</plist>
```

### Reference from pbxproj

Add `CODE_SIGN_ENTITLEMENTS = <YourApp>/<YourApp>.entitlements;` to BOTH Debug and Release `buildSettings` of your app target.

### Provisioning profile must include the capabilities

The wildcard `iOS Team Provisioning Profile: *` does NOT include these capabilities by default. **xcodebuild CLI cannot add them** — Apple requires Apple ID login + Xcode GUI to register the capability with the App ID and re-issue a provisioning profile.

**One-time setup (Xcode GUI):**
1. Open the workspace in Xcode
2. Settings → Accounts → "+" → Add Apple ID associated with your team
3. Select target → Signing & Capabilities tab → enable "Automatically manage signing"
4. Click `+ Capability` and add: **Increased Memory Limit**, **Extended Virtual Addressing**, **Increased Debugging Memory Limit**
5. Xcode auto-creates a new App ID + provisioning profile with the capabilities

After this once-per-machine step, `xcodebuild -allowProvisioningUpdates ...` from CLI works.

### Verification on device

A successful run shows in the LiteRT-LM stderr:
```
RunPrefillAsync status: OK
RunDecodeAsync
```
Failure (missing entitlements) shows:
```
ERROR: [litert_compiled_model.cc:162] Failed to allocate tensors
RunPrefillAsync status: INTERNAL: ERROR: ... Failed to invoke the compiled model
Memory warning (TRIM_MEMORY_RUNNING_CRITICAL) received by JS VM
```

## Troubleshooting

### `Engine type not found: 1` at engine_create

The static initializers `LITERT_LM_REGISTER_ENGINE` (in `engine_impl.o`)
got dead-stripped. Verify Podfile post_install has the
`-Wl,-force_load,...LiteRTLMInit.a` flag, and that
`LiteRTLMInit.a` exists inside the xcframework's framework dir.

### `GPU accelerator could not be loaded` in stderr

`gpu_registry.cc` does `dlopen("libLiteRtMetalAccelerator.dylib")` with
the *plain* library name. Verify the Run Script copied the dylib into
`App.app/Frameworks/libLiteRtMetalAccelerator.dylib` (not nested inside a
`.framework` bundle), and that codesign succeeded for both the dylib and
the app.

### `Code signing identifier does not match bundle identifier`

For a dylib re-signed by the Run Script: pass an explicit
`--identifier "com.your.bundle.id"` to `codesign`. The default identifier
matches the dylib filename, which won't match the framework's expected
bundle id.

### Hermes "Open debugger to view warnings" / `tool_call=null`

Confirmed Hermes regex bug: `parseGemma4ToolCall` uses indexOf+slice
instead of regex (commit f87e9c). If you forked an older state, pull the
latest and rebuild the JS bundle (metro `--reset-cache`).

### `Failed to allocate tensors` on device (sim works fine)

You are missing the iOS memory entitlements. See the **iOS memory entitlements**
section above — without `com.apple.developer.kernel.increased-memory-limit`
the OS jetsam cap kills tensor allocation regardless of CPU/GPU backend
choice.

### `nanov2_guard_corruption_detected` on engine_create

This was the symptom of the previous "fundamental wall" misdiagnosis.
On main HEAD with this build script's flags
(`litert_link_capi_so=true` + `resolve_symbols_in_exec=false`), the
archive's absl uses `lts_20260107` inline namespace and the dylibs use
no prefix — so they *don't* collide. If you still see this crash:
- Check archive's absl namespace: `nm libengine-*.a | grep "lts_" | head`
- Check dylib's: `nm libLiteRt.dylib | grep "lts_" | head` (should be empty)
- If both have the same namespace prefix you have an ABI mismatch.

### `Unable to install vendored xcframework because it contains dynamic libraries`

Cocoapods doesn't allow `*.dylib` files inside `vendored_frameworks`. Wrap
the dylib in a minimal `.framework` directory + Info.plist (already done
for `GemmaModelConstraintProvider.xcframework`), or treat it as a
runtime-only `dlopen` target and stage it in `ios/dylibs/` for the Run
Script to copy.

## Reference WORKSPACE pin

`/tmp/LiteRT-LM/WORKSPACE` (main HEAD `60ae696`):

```python
LITERT_REF = "47615eb6eaec25e8dfcd1aba922c560a57cba0a2"  # 2026-04-27
LITERT_SHA256 = "1d198ae395ba47d64dec282602de56b568ea964963861451933f00c6a39fbf2d"
TENSORFLOW_REF = "49e7f1937d1509dd7fea41bff9ccc994baa97258"
```

The prebuilt dylibs in `prebuilt/ios_arm64/` and `prebuilt/ios_sim_arm64/`
were compiled from the same `LITERT_REF`, which is why the wrapper archive
we build here is ABI-compatible with them.

## Function-calling reliability findings (Gemma 4 E2B autonomous mode)

After 30+ empirical iterations on real device + simulator, the following
boundaries of what Gemma 4 E2B (the model on the public HuggingFace
`litert-community/gemma-4-E2B-it-litert-lm` release, identical file used by
Google AI Edge Gallery) can and cannot do for autonomous function calling:

### What works reliably
- **Rule 1 explicit FC** for tools whose name does NOT trigger Gemma 4 RLHF
  refusal patterns. `echo`, `add_numbers` work consistently when the user
  message contains "Use the X tool to ..." pattern.
- **General chat** (Rule 3) — model answers from knowledge.
- **Honest unknown + suggest follow-up** (Rule 4).

### What does NOT work
- **Autonomous Rule 2** ("the user asked for real-time / fact-grounded
  data, model should call the tool"). Gemma 4 E2B's RLHF refuses with "I do
  not have access to real-time data" regardless of:
  - 5+ different system prompts (including verbatim Edge Gallery
    `defaultSystemPrompt` MUST-execute pattern)
  - Tool description rewrites (Anthropic-style "use this when..." +
    anti-apology language)
  - User query phrasing (4 variants: "right now", "Get the temperature",
    "Lookup", "Fetch the weather data")
  - Tool name rewrites (`get_weather` → `fetch_data`)
  - Parameter name (`city` → `location` matching Google's official example)
  - Decoding mode (`enableConstrainedDecoding` true and false)
  - Sampling (`temperature: 0` + `topK: 1` = pure greedy)
  - In-context few-shot via wrapper-level `priorMessages` injection
    (positive examples; self-correction examples — both backfire)
- **Rule 1 explicit `get_weather`** also degrades after a few scenarios in
  the same conversation — model says "I do not have access to a get_weather
  tool" even when user says "Use the get_weather tool".

### Why Edge Gallery succeeds with the same model

Edge Gallery does NOT do autonomous Rule 2 FC with Gemma 4 E2B. Source code
evidence (`AgentChatTaskModule.kt`, `AgentTools.kt` on
`google-ai-edge/gallery` `main` branch):
- Their FC tasks always run with `enableConversationConstrainedDecoding =
  true` AND a structurally-forced `load_skill` meta-tool pattern. The
  default system prompt mandates "you MUST call `load_skill`" as step 2 of
  the protocol. The model never decides "should I use a tool" — it always
  loads a skill first.
- Their FC-only "Mobile Actions" demo uses a completely different model:
  `litert-community/functiongemma-270m-ft-mobile-actions` (FC fine-tuned),
  not Gemma 4 E2B.

### SDK design implications

For an SDK targeting on-device Gemma 4 FC via this fork:
1. Document the autonomous Rule 2 limitation; do not promise it works.
2. Recommend Rule 1 explicit pattern with neutral tool names. Avoid
   "weather" and other RLHF-trigger themes.
3. For weather/real-time use cases, recommend FunctionGemma 270M
   (`litert-community/functiongemma-270m-ft-mobile-actions`) as a separate
   FC-only model.
4. To match Edge Gallery's reliability, customers must implement a
   `load_skill`-style meta-tool architecture; raw direct tools won't
   produce reliable autonomous selection.

### Wrapper config supporting in-context history
The `LLMConfig.priorMessages` field (added in this fork) accepts a JSON
array of prior user/assistant/tool turns and passes them as the
`messages_json` argument to `litert_lm_conversation_config_create`. This
enables in-context few-shot patterns. **Caveat**: pure positive examples
of get_weather use in priorMessages did NOT produce R2 success in our
tests; positive priorMessages even appears to CORRUPT tool visibility for
tools that are heavily referenced in the history. Self-correction patterns
(showing wrong-then-right) backfire because the model copies the wrong
example.
