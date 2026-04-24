# react-native-litert-lm

High-performance on-device LLM inference for React Native, powered by [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [Nitro Modules](https://github.com/mrousavy/nitro). Optimized for **Gemma 4** and other on-device language models.

## Features

- 🚀 **Native Performance** — Kotlin (Android) / C++ (iOS) via Nitro Modules JSI bindings
- 🧠 **Gemma 4 Ready** — First-class support for Gemma 4 E2B/E4B multimodal models (text + vision + audio)
- ⚡ **GPU Acceleration** — GPU delegate (Android), Metal/MPS (iOS)
- 🔄 **Streaming Support** — Token-by-token generation callbacks
- 📱 **Cross-Platform** — Android API 26+ / iOS 15.0+
- 🖼️ **Multimodal** — Image and audio input support
- 🧵 **Async API** — Non-blocking inference on dedicated large-stack threads
- 📊 **Real Memory Tracking** — OS-level memory metrics (RSS, native heap, available memory) via native APIs
- 🧮 **Zero-Copy Buffers** — Memory snapshots stored in native ArrayBuffers via Nitro Modules
- 📥 **Automatic Model Download** — Downloads models from URL with progress tracking and local caching

## Installation

```bash
npm install react-native-litert-lm react-native-nitro-modules
```

### Expo

Add to your `app.json`:

```json
{
  "expo": {
    "plugins": ["react-native-litert-lm"],
    "android": {
      "minSdkVersion": 26
    }
  }
}
```

Then create a development build:

```bash
npx expo prebuild
npx expo run:android  # Android
npx expo run:ios      # iOS
```

> **Note**: Only ARM devices/simulators are supported. x86_64 Android emulators are not supported.

### Bare React Native

```bash
# Android
cd android && ./gradlew clean

# iOS
cd ios && pod install
```

## Example App

The `example/` directory contains a fully functional test app with a dark-themed diagnostic UI that demonstrates:

- Model downloading with progress tracking
- Text inference (blocking and streaming)
- Multi-turn conversation with context retention
- Performance benchmarking (tokens/sec, latency)
- Real-time memory tracking
- Quick chat interface

### Running the Example

1. **Build the library** (compiles TypeScript to `lib/`):

   ```bash
   npm run build
   ```

2. **Install example dependencies:**

   ```bash
   cd example
   npm install
   ```

   > **Important:** Use `npm` (not `pnpm`) for the example app. The library is linked via `file:..`, and npm creates a symlink so the iOS XCFramework is visible to CocoaPods. pnpm copies files based on the `files` array and misses `ios/Frameworks/`.

3. **Create a development build and run:**

   ```bash
   npx expo prebuild --clean
   npx expo run:android  # Android
   npx expo run:ios      # iOS (requires XCFramework — see "Building the iOS Engine" below)
   ```

> **Note:** If you change native code (C++/Kotlin/Obj-C++), you must run `npx expo prebuild --clean` again before rebuilding.

## Model Management

LiteRT-LM models (like Gemma 4) are large files (2–4 GB) and cannot be bundled into your app binary. They are downloaded at runtime.

### Automatic Downloading

The library handles downloading automatically when you pass a URL to `loadModel` or `useModel`. Downloads include:

- **Progress tracking** — real-time download percentage via callbacks
- **Local caching** — downloaded models are cached and reused across app launches
  - **Android**: app-local temp directory
  - **iOS**: `Library/Caches/litert_models/` (survives app relaunch; reclaimable by iOS under storage pressure)
- **HTTPS enforcement** — only secure URLs are accepted

### Manual Downloading (Optional)

If you prefer to manage downloads yourself (e.g., using `expo-file-system`), download the `.litertlm` file to a local path and pass that path to the library:

```typescript
import * as FileSystem from "expo-file-system";
import { GEMMA_4_E2B_IT } from "react-native-litert-lm";

const localPath = `${FileSystem.documentDirectory}gemma-4-E2B-it.litertlm`;

async function downloadModel() {
  const info = await FileSystem.getInfoAsync(localPath);
  if (info.exists) return localPath;

  await FileSystem.downloadAsync(GEMMA_4_E2B_IT, localPath);
  return localPath;
}
```

## Usage

### React Hook (Recommended)

The `useModel` hook manages the full model lifecycle: downloading, loading, inference, and cleanup.

```typescript
import { useModel, GEMMA_4_E2B_IT } from "react-native-litert-lm";
import { Platform } from "react-native";

function App() {
  const {
    model,
    isReady,
    downloadProgress,
    error,
    load,          // Manually trigger load
    deleteModel,   // Delete cached model file
    memorySummary, // Auto-updated memory stats (if tracking enabled)
  } = useModel(GEMMA_4_E2B_IT, {
    backend: 'cpu',
    autoLoad: true, // Default: true. Set false to load manually via load().
    systemPrompt: "You are a helpful assistant.",
    enableMemoryTracking: true,
  });

  if (!isReady) {
    return <Text>Loading... {Math.round(downloadProgress * 100)}%</Text>;
  }

  const generate = async () => {
    const response = await model.sendMessage("Hello!");
    console.log(response);
  };

  return <Button title="Generate" onPress={generate} />;
}
```

### Manual Usage

```typescript
import { createLLM } from "react-native-litert-lm";

const llm = createLLM();

// Load a model from URL (auto-downloads) or local path
await llm.loadModel("https://example.com/model.litertlm", {
  backend: "gpu",
  systemPrompt: "You are a helpful assistant.",
});

// Generate a response
const response = await llm.sendMessage("What is the capital of France?");
console.log(response);

// Clean up
llm.close();
```

### Streaming Generation

```typescript
llm.sendMessageAsync("Tell me a story", (token, done) => {
  process.stdout.write(token);
  if (done) console.log("\n--- Done ---");
});
```

### Multimodal (Image / Audio)

> **Note**: Multimodal is fully supported on Android. iOS has the code paths implemented but vision/audio executors may not be available in the current XCFramework build — use `checkMultimodalSupport()` to verify at runtime.

```typescript
import { checkMultimodalSupport } from "react-native-litert-lm";

const warning = checkMultimodalSupport();
if (warning) {
  console.warn(warning); // Experimental on iOS
} else {
  // Image input (for vision models like Gemma 4)
  // Images >1024px are automatically resized to prevent OOM
  const response = await llm.sendMessageWithImage(
    "What's in this image?",
    "/path/to/image.jpg",
  );

  // Audio input
  const transcription = await llm.sendMessageWithAudio(
    "Transcribe this audio",
    "/path/to/audio.wav",
  );
}
```

### Performance Stats

```typescript
const stats = llm.getStats();
console.log(`Generated ${stats.completionTokens} tokens`);
console.log(`Speed: ${stats.tokensPerSecond.toFixed(1)} tokens/sec`);
console.log(`Time to first token: ${stats.timeToFirstToken.toFixed(0)} ms`);
```

### Memory Tracking

The library provides real OS-level memory data — no estimation. It reads directly from `mach_task_basic_info` (iOS) and `Debug.getNativeHeapAllocatedSize()` + `/proc/self/status` (Android).

#### Direct Memory Query

```typescript
const usage = llm.getMemoryUsage();
console.log(
  `Native heap: ${(usage.nativeHeapBytes / 1024 / 1024).toFixed(1)} MB`,
);
console.log(`RSS: ${(usage.residentBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(
  `Available: ${(usage.availableMemoryBytes / 1024 / 1024).toFixed(1)} MB`,
);
console.log(`Low memory: ${usage.isLowMemory}`);
```

#### Automatic Tracking with Native Buffers

Enable memory tracking to automatically record snapshots in a native-backed `ArrayBuffer` after every inference call:

```typescript
const llm = createLLM({
  enableMemoryTracking: true,
  maxMemorySnapshots: 256,
});

await llm.loadModel("/path/to/model.litertlm", { backend: "cpu" });
await llm.sendMessage("Hello!");

const summary = llm.memoryTracker!.getSummary();
console.log(
  `Peak RSS: ${(summary.peakResidentBytes / 1024 / 1024).toFixed(1)} MB`,
);
console.log(
  `RSS Delta: ${(summary.residentDeltaBytes / 1024 / 1024).toFixed(1)} MB`,
);
```

#### Using `useModel` with Memory Tracking

```typescript
const { model, isReady, memorySummary } = useModel(modelUrl, {
  enableMemoryTracking: true,
  maxMemorySnapshots: 100,
});

// memorySummary auto-updates after each inference call
if (memorySummary) {
  console.log(`Current RSS: ${memorySummary.currentResidentBytes}`);
  console.log(`Peak RSS: ${memorySummary.peakResidentBytes}`);
}
```

#### Standalone Memory Tracker

```typescript
import {
  createMemoryTracker,
  createNativeBuffer,
} from "react-native-litert-lm";

const tracker = createMemoryTracker(100);

tracker.record({
  timestamp: Date.now(),
  nativeHeapBytes: 50_000_000,
  residentBytes: 200_000_000,
  availableMemoryBytes: 4_000_000_000,
});

// Access the underlying native buffer (zero-copy transfer to native code)
const buffer = tracker.getNativeBuffer();
```

## Supported Models

Download `.litertlm` models automatically using the exported URL constants, or manually from [HuggingFace](https://huggingface.co/litert-community):

| Constant               | Model                              | Size     | Min RAM | Auth Required |
| :--------------------- | :--------------------------------- | :------- | :------ | :------------ |
| `GEMMA_4_E2B_IT`       | Gemma 4 E2B (Multimodal, IT)       | 2.58 GB  | 4 GB+   | ❌ No          |
| `GEMMA_4_E4B_IT`       | Gemma 4 E4B (Higher Quality)       | 3.65 GB  | 6 GB+   | ❌ No          |
| `GEMMA_3N_E2B_IT_INT4` | Gemma 3n E2B (Int4, Multimodal)    | ~1.3 GB  | 4 GB+   | ✅ HuggingFace |

> **Recommended:** Use `GEMMA_4_E2B_IT` for most use cases. It's multimodal (text + vision + audio) and downloads directly from HuggingFace without requiring an account.
>
> **iOS Note:** Models larger than ~2 GB (like Gemma 4) require the `com.apple.developer.kernel.extended-virtual-addressing` entitlement. See [iOS Entitlements](#ios-entitlements) below.

**Other compatible models** (download manually from HuggingFace):

| Model         | Size    | Min RAM | Notes                 |
| ------------- | ------- | ------- | --------------------- |
| Gemma 3 1B    | ~1 GB   | 4 GB+   | Smallest, fastest     |
| Phi-4 Mini    | ~2 GB   | 4 GB+   | Microsoft's small LLM |
| Qwen 2.5 1.5B | ~1.5 GB | 4 GB+   | Multilingual          |

## API Reference

### `createLLM(options?): LiteRTLM`

Creates a new LLM inference engine instance.

- `options.enableMemoryTracking` — enable automatic memory snapshot recording
- `options.maxMemorySnapshots` — max number of snapshots to retain (default: 256)

### `loadModel(path, config?): Promise<void>`

Loads a model from a local path or HTTPS URL.

| Parameter             | Type     | Default | Description                               |
| --------------------- | -------- | ------- | ----------------------------------------- |
| `path`                | `string` | —       | Absolute path to `.litertlm` or HTTPS URL |
| `config.backend`      | `string` | `'cpu'` | `'cpu'`, `'gpu'`, or `'npu'`              |
| `config.systemPrompt` | `string` | —       | System prompt for the model               |
| `config.temperature`  | `number` | `0.7`   | Sampling temperature                      |
| `config.topK`         | `number` | `40`    | Top-K sampling                            |
| `config.topP`         | `number` | `0.95`  | Top-P (nucleus) sampling                  |
| `config.maxTokens`    | `number` | `1024`  | Maximum generation length                 |

#### Backend Options

| Backend | Engine              | Speed   | Notes                                          |
| ------- | ------------------- | ------- | ---------------------------------------------- |
| `'cpu'` | CPU inference       | Slowest | Always available, lower RAM requirement        |
| `'gpu'` | GPU / Metal         | Fast    | Recommended default                            |
| `'npu'` | NPU / Neural Engine | Fastest | Requires supported hardware; falls back to GPU |

> **iOS**: `'cpu'` is the recommended default backend. `'gpu'` (Metal/MPS) is also supported. The engine automatically tries multiple backend combinations if the primary one fails.

### `sendMessage(message): Promise<string>`

Runs inference synchronously on a background thread. Returns the complete response.

### `sendMessageAsync(message, callback)`

Streaming generation. Callback signature: `(token: string, isDone: boolean) => void`.

### `sendMessageWithImage(message, imagePath): Promise<string>`

Send a message with an image (for vision models like Gemma 4 E2B).

### `sendMessageWithAudio(message, audioPath): Promise<string>`

Send a message with audio (for audio-capable models like Gemma 4 E2B).

### `getStats(): GenerationStats`

Returns performance metrics from the last inference call.

```typescript
interface GenerationStats {
  tokensPerSecond: number;
  totalTime: number; // seconds
  timeToFirstToken: number; // seconds
  promptTokens: number;
  completionTokens: number;
  prefillSpeed: number; // tokens/sec
}
```

### `getMemoryUsage(): MemoryUsage`

Returns real OS-level memory usage.

```typescript
interface MemoryUsage {
  nativeHeapBytes: number;
  residentBytes: number;
  availableMemoryBytes: number;
  isLowMemory: boolean;
}
```

### `getHistory(): Message[]`

Returns the conversation history.

### `resetConversation()`

Clears conversation context and starts a fresh session.

### `close()`

Releases all native resources. Call when the model is no longer needed.

### `deleteModel(fileName): Promise<void>`

Deletes a cached model file from the app's local storage.

### Utility Functions

```typescript
import {
  checkBackendSupport,
  checkMultimodalSupport,
  getRecommendedBackend,
  applyGemmaTemplate,
  applyPhiTemplate,
  applyLlamaTemplate,
} from "react-native-litert-lm";

// Check if a backend is supported
const warning = checkBackendSupport("npu"); // string | undefined
const mmError = checkMultimodalSupport(); // string | undefined
const backend = getRecommendedBackend(); // 'gpu' | 'cpu'

// Manual prompt formatting (advanced)
const prompt = applyGemmaTemplate(
  [{ role: "user", content: "Hello!" }],
  "You are helpful.",
);
```

## Requirements

| Dependency                 | Version       |
| -------------------------- | ------------- |
| React Native               | 0.76+         |
| react-native-nitro-modules | 0.35.0+       |
| Android API                | 26+ (ARM64)   |
| iOS                        | 15.0+ (ARM64) |
| LiteRT-LM Engine            | 0.10.2          |

## Platform Support

| Platform | Status   | Architecture | Backends         |
| -------- | -------- | ------------ | ---------------- |
| Android  | ✅ Ready | arm64-v8a    | CPU, GPU, NPU    |
| iOS      | ✅ Ready | arm64        | CPU, GPU (Metal) |

### iOS Feature Matrix

| Feature                      | Status | Notes                                                 |
| ---------------------------- | ------ | ----------------------------------------------------- |
| Text inference (blocking)    | ✅     | Via LiteRT-LM C API                                   |
| Text inference (streaming)   | ✅     | Token-by-token callbacks                              |
| CPU inference                | ✅     | Recommended default backend                           |
| GPU inference (Metal/MPS)    | ✅     | Supported via `backend: 'gpu'`                        |
| Model download with progress | ✅     | NSURLSession, cached in `Caches/`                     |
| Memory tracking              | ✅     | `mach_task_basic_info`                                |
| Multi-turn conversation      | ✅     | Context retained across turns                         |
| Multimodal (image/audio)     | 🧪     | Code paths exist; vision/audio executors experimental |
| Constrained decoding         | ❌     | Requires llguidance Rust runtime                      |
| Function calling             | ❌     | Requires Rust CXX bridge runtime                      |

### iOS Entitlements

Models larger than ~2 GB (like Gemma 4 E2B at 2.58 GB) require the **Extended Virtual Addressing** entitlement on iOS physical devices. Without it, iOS limits virtual memory to ~2 GB and the app will be killed by Jetsam.

Add to your app's `.entitlements` file:

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

> **Note:** This entitlement requires a **paid Apple Developer account** ($99/year). Gemma 3n E2B (~1.3 GB) works without it.

## Building the iOS Engine

The iOS build uses a **Bazel-to-XCFramework pipeline** that compiles the LiteRT-LM C engine and all transitive dependencies into a static library (~82–84 MB).

### Prerequisites

- **Bazel 7.6.1+** (via [Bazelisk](https://github.com/bazelbuild/bazelisk) recommended)
- **Xcode command line tools** (`xcode-select --install`)

### Build

```bash
./scripts/build-ios-engine.sh
```

This will:

1. Clone/checkout LiteRT-LM `v0.10.2` source into `.litert-lm-build/`
2. Apply `scripts/patches/ios-engine-fixes.patch` (PromptTemplate simplification, linker fixes)
3. Build `//c:engine` for `ios_arm64` and `ios_sim_arm64` via Bazel
4. Collect all transitive `.o` files (engine, protobuf, re2, sentencepiece, etc.)
5. Compile C/C++ stubs for unavailable Rust dependencies
6. Merge ~1,909 object files into a static library via `libtool`
7. Package into `ios/Frameworks/LiteRTLM.xcframework`

### Output

```
ios/Frameworks/LiteRTLM.xcframework/
├── Info.plist
├── ios-arm64/LiteRTLM.framework/              # Device
│   ├── LiteRTLM                                # ~82 MB static library
│   └── Headers/litert_lm_engine.h
└── ios-arm64-simulator/LiteRTLM.framework/    # Simulator
    ├── LiteRTLM                                # ~84 MB static library
    └── Headers/litert_lm_engine.h
```

### FFI Stubs

Certain LiteRT-LM features depend on Rust libraries (llguidance, CXX bridge, MinijinjaTemplate) that are not available in the iOS Bazel build. These are replaced with stubs:

| Stub File                            | Location         | Purpose                                  |
| ------------------------------------ | ---------------- | ---------------------------------------- |
| `cxx_bridge_stubs.cc`                | `scripts/stubs/` | CXX bridge runtime + Rust FFI type stubs |
| `llguidance_stubs.c`                 | `scripts/stubs/` | llguidance constrained decoding C API    |
| `gemma_model_constraint_provider.cc` | `scripts/stubs/` | Gemma constraint provider factory        |

Additionally, `PromptTemplate` is patched at build time to use a simplified C++ template formatter instead of the Rust MinijinjaTemplate, which avoids all Rust FFI calls during conversation setup.

> **Text inference works fully without these Rust components.** Only constrained decoding, function calling parsers, and advanced Jinja2 template features are affected.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  React Native (TypeScript)                      │
│  useModel() / createLLM() / sendMessage()       │
├─────────────────────────────────────────────────┤
│  Nitro Modules JSI Bridge                       │
├──────────────────────┬──────────────────────────┤
│  Android (Kotlin)    │  iOS (C++)               │
│  HybridLiteRTLM.kt   │  HybridLiteRTLM.cpp      │
│  litertlm-android    │  LiteRTLM C API          │
│  AAR (GPU delegate)  │  XCFramework (Metal)     │
└──────────────────────┴──────────────────────────┘
```

- **Android**: Kotlin (`HybridLiteRTLM.kt`) interfacing with the `litertlm-android` AAR.
- **iOS**: C++ (`HybridLiteRTLM.cpp`) interfacing with the LiteRT-LM C API via a prebuilt `LiteRTLM.xcframework`. All engine operations (load, inference, streaming) run on dedicated `pthread` threads with 8 MB stack to accommodate XNNPack's stack requirements. Platform-specific code (model downloading, file management) is in Objective-C++ (`ios/IOSDownloadHelper.mm`).

> **For contributors**: Changes to `cpp/HybridLiteRTLM.cpp` do not affect Android. Feature changes must be applied to both the Kotlin and C++ implementations.

## License

The code in this repository is licensed under the **[MIT License](LICENSE)**.

### ⚠️ AI Model Disclaimer

This library is an execution engine for on-device LLMs. The AI models themselves are **not** distributed with this package and have their own licenses:

- **Gemma (Google)**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Llama 3 (Meta)**: [Llama 3.2 Community License](https://www.llama.com/llama3/license/)
- **Qwen (Alibaba)**: [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE)
- **Phi (Microsoft)**: [MIT License](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE)

By downloading and using these models, you agree to their respective licenses and acceptable use policies. The author of `react-native-litert-lm` takes no responsibility for model outputs or applications built with them.
