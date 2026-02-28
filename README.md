# react-native-litert-lm

High-performance LLM inference for React Native powered by [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [Nitro Module](https://github.com/mrousavy/nitro). Optimized for **Gemma 3n** and other on-device language models.

## Features

- 🚀 **Native Performance** - Kotlin (Android) / C++ (iOS) implementation via Nitro Modules
- 🧠 **Gemma 3n Ready** - First-class support for Gemma 3n E2B/E4B models
- ⚡ **GPU Acceleration** - GPU delegate (Android), Metal (iOS when available)
- 📦 **Bundled Tokenizer** - No separate tokenization library needed
- 🔄 **Streaming Support** - Token-by-token generation callbacks
- 📱 **Cross-Platform** - Android API 26+
- 🖼️ **Multimodal** - Image and audio input support (Android Beta, iOS coming soon)
- 🧵 **Async API** - Non-blocking inference to prevent UI freezes
- 📊 **Real Memory Tracking** - OS-level memory metrics (RSS, native heap, available memory) via native APIs
- 🧮 **Zero-Copy Buffers** - Memory snapshots stored in native ArrayBuffers via `NitroModules.createNativeArrayBuffer()` (v0.34+)

## Status

> ⚠️ **Early Preview**: This library is under active development. Android is functional with enough RAM, iOS implementation pending LiteRT-LM iOS release. Please report any issues on the [GitHub issues](https://github.com/hung-yueh/react-native-litert-lm/issues).

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
npx expo run:android
```

> **Note**: Only ARM devices are supported (physical devices or ARM emulators). x86_64 emulators are not supported.

### Bare React Native

```bash
cd android && ./gradlew clean
cd ios && pod install  # iOS coming soon
```

## Example App

The repository includes a fully functional example app in the `example/` directory with a dark-themed diagnostic UI that demonstrates model loading, inference, memory tracking, and performance stats.

To run it:

1.  **Build the library** (compiles TypeScript to `lib/`):

    ```bash
    npm run build
    ```

2.  **Navigate to the example directory and install dependencies:**

    ```bash
    cd example
    npm install
    ```

3.  **Create a development build and run on Android:**
    ```bash
    npx expo prebuild --clean
    npx expo run:android
    ```

> **Note:** If you change native code (C++/Kotlin), you must run `npx expo prebuild --clean` again.

## Model Management

LiteRT-LM models (like Gemma 3n) are large files (3GB+) and cannot be bundled directly into your app's binary. You must download them at runtime to a writable directory (e.g., `DocumentDirectory`).

### Automatic Downloading

The library supports automatic downloading when you pass a URL to `loadModel` or `useModel`.

### Manual Downloading (Optional)

If you prefer to manage downloads manually (e.g., using `rn-fetch-blob` or `expo-file-system`), you can download the file to a local path and pass that path to the library.

```typescript
import { FileSystem } from "react-native-file-access";
// or import * as FileSystem from 'expo-file-system';

const MODEL_URL =
  "https://huggingface.co/litert-community/gemma-3n-2b-it/resolve/main/model.litertlm";
const localPath = `${FileSystem.DocumentDirectoryPath}/gemma-3n.litertlm`;

async function downloadModel() {
  if (await FileSystem.exists(localPath)) return localPath;

  // Download logic here...
  return localPath;
}
```

## Usage

### React Hook (Recommended)

The `useModel` hook manages the model lifecycle, including downloading, loading, and unloading.

```typescript
import { useModel, GEMMA_3N_E2B_IT_INT4 } from "react-native-litert-lm";

function App() {
  const {
    model,
    isReady,
    downloadProgress,
    load,   // Manually trigger load
    deleteModel // Delete model file
  } = useModel(
    GEMMA_3N_E2B_IT_INT4,
    {
      backend: "cpu",
      autoLoad: true, // Default: true. Set false to load manually.
      systemPrompt: "You are a helpful assistant."
    }
  );

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

// Generate response (async)
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

### Multimodal (Image/Audio)

```typescript
import { checkMultimodalSupport } from "react-native-litert-lm";

// Check platform support first
const error = checkMultimodalSupport();
if (error) {
  console.warn(error); // iOS not yet supported
} else {
  // Image input (for vision models like Gemma 3n)
  // Images >1024px are automatically resized to prevent OOM
  const response = await llm.sendMessageWithImage(
    "What's in this image?",
    "/path/to/image.jpg",
  );

  // Audio input (for audio models)
  const transcription = await llm.sendMessageWithAudio(
    "Transcribe this audio",
    "/path/to/audio.wav",
  );
}
```

### Check Performance

```typescript
const stats = llm.getStats();
console.log(`Generated ${stats.completionTokens} tokens`);
console.log(`Speed: ${stats.tokensPerSecond.toFixed(1)} tokens/sec`);
```

### Memory Tracking

The library provides real OS-level memory usage data. You can query memory at any time, or enable automatic tracking to record snapshots after each inference call.

#### Direct Memory Query

```typescript
// Get a single real-time snapshot from native APIs
const usage = llm.getMemoryUsage();
console.log(`Native heap: ${(usage.nativeHeapBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(`RSS: ${(usage.residentBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(`Available: ${(usage.availableMemoryBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(`Low memory: ${usage.isLowMemory}`);
```

#### Automatic Tracking with Native Buffers

Enable memory tracking to automatically record snapshots in a native-backed `ArrayBuffer` (allocated via `NitroModules.createNativeArrayBuffer()`) after every inference call:

```typescript
import { createLLM } from 'react-native-litert-lm';

const llm = createLLM({
  enableMemoryTracking: true,
  maxMemorySnapshots: 256, // default
});

await llm.loadModel('/path/to/model.litertlm', { backend: 'cpu' });
await llm.sendMessage('Hello!');

// Review tracked data
const summary = llm.memoryTracker!.getSummary();
console.log(`Peak RSS: ${(summary.peakResidentBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(`Peak Native Heap: ${(summary.peakNativeHeapBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(`RSS Delta: ${(summary.residentDeltaBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(`Snapshots: ${summary.snapshotCount}`);
```

#### Using the `useModel` Hook with Memory Tracking

```typescript
import { useModel } from 'react-native-litert-lm';

const { model, isReady, memorySummary, memoryTracker } = useModel(modelUrl, {
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
import { createMemoryTracker, createNativeBuffer } from 'react-native-litert-lm';

// Create a tracker backed by a native ArrayBuffer
const tracker = createMemoryTracker(100);

// Manually record snapshots
tracker.record({
  timestamp: Date.now(),
  nativeHeapBytes: 50_000_000,
  residentBytes: 200_000_000,
  availableMemoryBytes: 4_000_000_000,
});

// Access the underlying native buffer (for zero-copy transfer to native code)
const buffer = tracker.getNativeBuffer();

// Create a standalone native buffer for custom use
const customBuffer = createNativeBuffer(1024);
```

## Supported Models

Download `.litertlm` models automatically using the exported constants or from [HuggingFace](https://huggingface.co/litert-community):

| Model Constant         | Description                            | Size | Min Device RAM |
| :--------------------- | :------------------------------------- | :--- | :------------- |
| `GEMMA_3N_E2B_IT_INT4` | Gemma 3n E2B (Instruction Tuned, Int4) | ~3GB | 4GB+           |

| Other Models  | Size   | Min Device RAM | Use Case              |
| ------------- | ------ | -------------- | --------------------- |
| Gemma 3n E4B  | ~4GB   | 8GB+           | Higher quality        |
| Gemma 3 1B    | ~1GB   | 4GB+           | Smallest, fastest     |
| Phi-4 Mini    | ~2GB   | 4GB+           | Microsoft's small LLM |
| Qwen 2.5 1.5B | ~1.5GB | 4GB+           | Multilingual          |

## API Reference

### `createLLM(): LiteRTLM`

Creates a new LLM inference engine instance.

### `loadModel(path, config?): Promise<void>`

- `path: string` - Absolute path to `.litertlm` file OR a public URL (http/https). If a URL is provided, the model will be downloaded automatically.
- `config.systemPrompt` - System prompt to guide model behavior (e.g., "You are a helpful assistant.")
- `config.backend` - `'cpu'` | `'gpu'` | `'npu'` (default: `'gpu'`)
- `config.temperature` - Sampling temperature (default: 0.7)
- `config.topK` - Top-K sampling (default: 40)
- `config.maxTokens` - Max generation length (default: 1024)

> **Note**: Vision encoder is always set to GPU (required by Gemma 3n). Audio encoder is always set to CPU (optimal for audio).

#### Backend Options

| Backend | Description       | Speed   | Compatibility                              |
| ------- | ----------------- | ------- | ------------------------------------------ |
| `'cpu'` | CPU inference     | Slowest | Always available with less RAM requirement |
| `'gpu'` | GPU acceleration  | Fast    | Recommended default                        |
| `'npu'` | NPU/Neural Engine | Fastest | Requires supported hardware                |

> ⚠️ **NPU Note**: NPU acceleration requires compatible hardware (Qualcomm Hexagon, MediaTek APU, etc.). If unavailable, LiteRT-LM automatically falls back to GPU.

### `sendMessage(message): Promise<string>`

Blocking generation (executed on background thread). Returns complete response.

### `sendMessageAsync(message, callback)`

Streaming generation. Callback receives `(token, isDone)`.

### `sendMessageWithImage(message, imagePath): Promise<string>`

Send a message with an image attachment (for vision models).

### `sendMessageWithAudio(message, audioPath): Promise<string>`

Send a message with an audio attachment (for audio models).

### `getMemoryUsage(): MemoryUsage`

Returns real OS-level memory usage statistics from native APIs. No estimation — reads directly from `mach_task_basic_info` (iOS) / `Debug.getNativeHeapAllocatedSize()` + `/proc/self/status` (Android).

```typescript
interface MemoryUsage {
  nativeHeapBytes: number;      // Native heap allocated bytes
  residentBytes: number;        // Process RSS in bytes
  availableMemoryBytes: number; // Available system memory in bytes
  isLowMemory: boolean;         // Whether the system considers memory low
}
```

### `getHistory(): Message[]`

Get conversation history.

### `resetConversation()`

Clear context and start fresh.

### `close()`

Release all native resources.

### `deleteModel(fileName): Promise<void>`

Deletes a model file from the app's internal storage and cleans up the engine instance.

### `getRecommendedBackend(): Backend`

Returns the recommended backend for the current platform (usually `'gpu'`).

### `checkBackendSupport(backend): string | undefined`

Returns a warning message if the specified backend may have issues on the current platform, or `undefined` if OK.

```typescript
import { checkBackendSupport } from "react-native-litert-lm";

const warning = checkBackendSupport("npu");
if (warning) {
  console.warn(warning);
}
```

### `checkMultimodalSupport(): string | undefined`

Returns an error message if multimodal (image/audio) is not supported on the current platform, or `undefined` if OK.

```typescript
import { checkMultimodalSupport } from "react-native-litert-lm";

const error = checkMultimodalSupport();
if (error) {
  console.warn(error); // iOS multimodal not yet supported
}
```

### Prompt Templates

For advanced use cases where you need to manually format prompts:

```typescript
import {
  applyGemmaTemplate,
  applyPhiTemplate,
  applyLlamaTemplate,
  ChatMessage,
} from "react-native-litert-lm";

const history: ChatMessage[] = [
  { role: "user", content: "Hello!" },
  { role: "model", content: "Hi there!" },
  { role: "user", content: "Tell me a joke" },
];

// For Gemma models
const gemmaPrompt = applyGemmaTemplate(history, "You are a comedian.");

// For Phi models
const phiPrompt = applyPhiTemplate(history);

// For Llama models
const llamaPrompt = applyLlamaTemplate(history, "You are helpful.");
```

## Requirements

- React Native 0.76+
- react-native-nitro-modules **0.34.1+** (required for `createNativeArrayBuffer` and memory tracking)
- Android API 26+ (ARM64 only)
- **LiteRT-LM Android SDK**: `0.9.0-alpha01` (bundled automatically)
- iOS 15.0+ (coming soon)

## Platform Support

| Platform | Status   | Architecture |
| -------- | -------- | ------------ |
| Android  | ✅ Ready | arm64-v8a    |
| iOS      | 🚧 Stub  | -            |

## Architecture

This library uses a split implementation strategy to maximize performance and compatibility:

- **Android**: Uses **Kotlin** (`HybridLiteRTLM.kt`) to interface directly with the `litertlm-android` AAR.
- **iOS**: Uses **C++** (`HybridLiteRTLM.cpp`) which will interface with the LiteRT-LM C++ headers (once released).

> **Note for Contributors**: Changes made to the C++ implementation (`cpp/`) **do not** affect Android. You must apply feature changes to both the Kotlin and C++ implementations.

## License

The code in this repository is licensed under the **[MIT License](LICENSE)**.

### ⚠️ Important AI Model Disclaimer

This library acts as an execution engine for On-Device Large Language Models (LLMs). The AI models themselves are **not** distributed with this package and are **not** covered by the MIT license.

By downloading and running these models within your app, you agree to comply with their respective licenses and acceptable use policies:

- **Gemma (Google)**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Llama 3 (Meta)**: [Llama 3.2 Community License](https://www.llama.com/llama3/license/)
- **Qwen (Alibaba)**: [Apache 2.0 License](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE)
- **Phi (Microsoft)**: [MIT License](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE)

_The author of `react-native-litert-lm` takes no responsibility for the outputs generated by these models or the applications built using them._
