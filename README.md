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

## Model Management

LiteRT-LM models (like Gemma 3n) are large files (3GB+) and cannot be bundled directly into your app's binary. You must download them at runtime to a writable directory (e.g., `DocumentDirectory`).

### Downloading Models

We recommend using `rn-fetch-blob` or `expo-file-system` to download models.

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

### Basic Generation

```typescript
import { createLLM } from "react-native-litert-lm";

const llm = createLLM();

// Load a Gemma 3n model (async)
await llm.loadModel("/path/to/gemma-3n-e2b.litertlm", {
  backend: "gpu",
  temperature: 0.7,
  maxTokens: 512,
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
// Image input (for vision models like Gemma 3n)
// ⚠️ Ensure model is loaded with { maxTokens: 1024+ }
const response = await llm.sendMessageWithImage(
  "What's in this image?",
  "/path/to/image.jpg",
);

// Audio input (for audio models)
const transcription = await llm.sendMessageWithAudio(
  "Transcribe this audio",
  "/path/to/audio.wav",
);
```

### Check Performance

```typescript
const stats = llm.getStats();
console.log(`Generated ${stats.completionTokens} tokens`);
console.log(`Speed: ${stats.tokensPerSecond.toFixed(1)} tokens/sec`);
```

## Supported Models

Download `.litertlm` models from [HuggingFace](https://huggingface.co/litert-community):

| Model         | Size   | Min Device RAM | Use Case                  |
| ------------- | ------ | -------------- | ------------------------- |
| Gemma 3n E2B  | ~3GB   | 4GB+           | Efficient, fast responses |
| Gemma 3n E4B  | ~4GB   | 8GB+           | Higher quality            |
| Gemma 3 1B    | ~1GB   | 4GB+           | Smallest, fastest         |
| Phi-4 Mini    | ~2GB   | 4GB+           | Microsoft's small LLM     |
| Qwen 2.5 1.5B | ~1.5GB | 4GB+           | Multilingual              |

## API Reference

### `createLLM(): LiteRTLM`

Creates a new LLM inference engine instance.

### `loadModel(path, config?): Promise<void>`

- `path: string` - Absolute path to `.litertlm` file
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

### `getHistory(): Message[]`

Get conversation history.

### `resetConversation()`

Clear context and start fresh.

### `close()`

Release all native resources.

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

## Requirements

- React Native 0.76+
- react-native-nitro-modules 0.33.2+
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
