import { NitroModules } from "react-native-nitro-modules";
import { LiteRTLM, LLMConfig } from "./specs/LiteRTLM.nitro";
import { createMemoryTracker, MemoryTracker } from "./memoryTracker";

/**
 * Extended LiteRT-LM instance with optional memory tracking and
 * augmented loadModel that accepts a download progress callback.
 */
export type LiteRTLMInstance = Omit<LiteRTLM, "loadModel"> & {
  memoryTracker?: MemoryTracker;
  loadModel: (
    pathOrUrl: string,
    config?: LLMConfig,
    onDownloadProgress?: (progress: number) => void,
  ) => Promise<void>;
};

/**
 * Creates a new LiteRT-LM inference engine instance.
 *
 * Optionally creates a native-backed memory tracker using
 * `NitroModules.createNativeArrayBuffer()` (v0.35+) for efficient
 * zero-copy memory usage tracking.
 *
 * @param options.enableMemoryTracking Enable automatic memory tracking (default: false)
 * @param options.maxMemorySnapshots Maximum number of memory snapshots to store (default: 256)
 */
export function createLLM(options?: {
  enableMemoryTracking?: boolean;
  maxMemorySnapshots?: number;
}): LiteRTLMInstance {
  const native = NitroModules.createHybridObject<LiteRTLM>("LiteRTLM");

  const enableTracking = options?.enableMemoryTracking ?? false;
  const tracker = enableTracking
    ? createMemoryTracker(options?.maxMemorySnapshots ?? 256)
    : undefined;

  /**
   * Record a real memory snapshot using OS-level APIs via getMemoryUsage().
   */
  const recordMemorySnapshot = () => {
    if (!tracker) return;
    try {
      const usage = native.getMemoryUsage();
      tracker.record({
        timestamp: Date.now(),
        nativeHeapBytes: usage.nativeHeapBytes,
        residentBytes: usage.residentBytes,
        availableMemoryBytes: usage.availableMemoryBytes,
      });
    } catch {
      // Ignore errors during memory tracking - it's non-critical
    }
  };

  return {
    ...native,
    memoryTracker: tracker,
    loadModel: async (
      pathOrUrl: string,
      config?: LLMConfig,
      onDownloadProgress?: (progress: number) => void,
    ) => {
      let modelPath = pathOrUrl;

      // Check if it's a URL — enforce HTTPS for model downloads
      if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
        if (pathOrUrl.startsWith("http://")) {
          throw new Error(
            "Insecure HTTP URLs are not allowed for model downloads. " +
              "Use HTTPS instead: " +
              pathOrUrl.replace("http://", "https://"),
          );
        }

        // Extract filename from URL
        const fileName = pathOrUrl.split("/").pop();
        if (!fileName) {
          throw new Error(`Invalid model URL: ${pathOrUrl}`);
        }

        console.log(`Checking model at ${pathOrUrl}...`);
        modelPath = await native.downloadModel(
          pathOrUrl,
          fileName,
          (progress) => {
            onDownloadProgress?.(progress);
          },
        );
        console.log(`Model downloaded to: ${modelPath}`);
      }

      const result = await native.loadModel(modelPath, config);

      // Record initial memory snapshot after model load
      if (tracker) {
        tracker.reset();
        recordMemorySnapshot();
      }

      return result;
    },
    sendMessage: async (...args: Parameters<typeof native.sendMessage>) => {
      const result = await native.sendMessage(...args);
      recordMemorySnapshot();
      return result;
    },
    sendMessageAsync: (...args: Parameters<typeof native.sendMessageAsync>) => {
      const [message, onToken] = args;
      native.sendMessageAsync(message, (token, done) => {
        onToken(token, done);
        if (done) {
          recordMemorySnapshot();
        }
      });
    },
    sendMessageWithImage: async (
      ...args: Parameters<typeof native.sendMessageWithImage>
    ) => {
      const result = await native.sendMessageWithImage(...args);
      recordMemorySnapshot();
      return result;
    },
    sendMessageWithAudio: async (
      ...args: Parameters<typeof native.sendMessageWithAudio>
    ) => {
      const result = await native.sendMessageWithAudio(...args);
      recordMemorySnapshot();
      return result;
    },
    getHistory: native.getHistory.bind(native),
    resetConversation: () => {
      native.resetConversation();
      // KV cache is cleared on reset, record the drop
      recordMemorySnapshot();
    },
    isReady: native.isReady.bind(native),
    getStats: native.getStats.bind(native),
    getMemoryUsage: native.getMemoryUsage.bind(native),
    close: native.close.bind(native),
    downloadModel: native.downloadModel.bind(native),
    deleteModel: native.deleteModel.bind(native),
  };
}
