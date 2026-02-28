import { useState, useEffect, useRef, useCallback } from "react";
import { LiteRTLM, LLMConfig } from "./index";
import { createLLM } from "./modelFactory";
import type { MemoryTracker, MemoryTrackerSummary } from "./memoryTracker";

export interface UseModelConfig extends LLMConfig {
  autoLoad?: boolean;
  /**
   * Enable memory tracking using native ArrayBuffers (v0.34+).
   * When enabled, memory usage is tracked after each inference call
   * using `NitroModules.createNativeArrayBuffer()` for zero-copy storage.
   * @default false
   */
  enableMemoryTracking?: boolean;
  /**
   * Maximum number of memory snapshots to store.
   * Each snapshot uses 32 bytes of native memory.
   * @default 256
   */
  maxMemorySnapshots?: number;
}

export interface UseModelResult {
  model: LiteRTLM | null;
  isReady: boolean;
  isGenerating: boolean;
  downloadProgress: number;
  error: string | null;
  generate: (prompt: string) => Promise<string>;
  reset: () => void;
  deleteModel: (fileName: string) => Promise<void>;
  load: () => Promise<void>;
  /**
   * Memory tracker instance (available when enableMemoryTracking is true).
   * Uses native ArrayBuffers allocated via `NitroModules.createNativeArrayBuffer()`
   * for efficient, zero-copy memory usage tracking.
   */
  memoryTracker: MemoryTracker | null;
  /**
   * Current memory tracking summary (null if tracking is disabled).
   * Updates automatically after each inference call.
   */
  memorySummary: MemoryTrackerSummary | null;
}

export function useModel(
  pathOrUrl: string,
  config?: UseModelConfig,
): UseModelResult {
  const modelRef = useRef<(LiteRTLM & { memoryTracker?: MemoryTracker }) | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [memorySummary, setMemorySummary] = useState<MemoryTrackerSummary | null>(null);

  // Extract autoLoad (default true) and memory tracking options
  const autoLoad = config?.autoLoad ?? true;
  const enableMemoryTracking = config?.enableMemoryTracking ?? false;
  const maxMemorySnapshots = config?.maxMemorySnapshots ?? 256;

  /**
   * Refresh memory summary from the tracker's native buffer.
   */
  const refreshMemorySummary = useCallback(() => {
    if (modelRef.current?.memoryTracker) {
      setMemorySummary(modelRef.current.memoryTracker.getSummary());
    }
  }, []);

  // Initialize the model instance
  useEffect(() => {
    modelRef.current = createLLM({
      enableMemoryTracking,
      maxMemorySnapshots,
    });
    let isMounted = true;

    // Cleanup on unmount
    return () => {
      isMounted = false;
      try {
        modelRef.current?.close();
      } catch (e) {
        console.warn("Failed to close model", e);
      }
    };
  }, [enableMemoryTracking, maxMemorySnapshots]);

  const load = useCallback(async () => {
    setIsReady(false);
    setError(null);
    setDownloadProgress(0);

    try {
      let modelPath = pathOrUrl;

      // Handle URL download manually to capture progress
      if (pathOrUrl.startsWith("http://") || pathOrUrl.startsWith("https://")) {
        const fileName = pathOrUrl.split("/").pop() || "model.bin";

        if (modelRef.current) {
          modelPath = await modelRef.current.downloadModel(
            pathOrUrl,
            fileName,
            (progress) => {
              setDownloadProgress(progress);
            },
          );
        }
      }

      if (modelRef.current) {
        // Create a clean config object for native loadModel (excluding autoLoad)
        const nativeConfig: LLMConfig = { ...config };
        delete (nativeConfig as any).autoLoad;

        await modelRef.current.loadModel(modelPath, nativeConfig);
        setIsReady(true);
      }
    } catch (e: any) {
      setError(e.message || "Failed to load model");
      console.error(e);
    }
  }, [pathOrUrl, config]);

  useEffect(() => {
    if (autoLoad) {
      load();
    }
  }, [autoLoad, load]);

  const generate = useCallback(
    async (prompt: string): Promise<string> => {
      if (!modelRef.current || !isReady) {
        throw new Error("Model not ready");
      }

      setIsGenerating(true);
      try {
        return new Promise<string>((resolve, reject) => {
          let fullResponse = "";
          try {
            modelRef.current?.sendMessageAsync(
              prompt,
              (token: string, done: boolean) => {
                fullResponse += token;
                if (done) {
                  refreshMemorySummary();
                  resolve(fullResponse);
                }
              },
            );
          } catch (e: any) {
            reject(e);
          }
        });
      } catch (e: any) {
        setError(e.message || "Generation failed");
        throw e;
      } finally {
        setIsGenerating(false);
      }
    },
    [isReady, refreshMemorySummary],
  );

  const reset = useCallback(() => {
    if (modelRef.current) {
      modelRef.current.resetConversation();
    }
  }, []);

  const deleteModel = useCallback(async (fileName: string): Promise<void> => {
    if (modelRef.current) {
      await modelRef.current.deleteModel(fileName);
      setIsReady(false);
      setDownloadProgress(0);
    }
  }, []);

  return {
    model: modelRef.current,
    isReady,
    isGenerating,
    downloadProgress,
    error,
    generate,
    reset,
    deleteModel,
    load,
    memoryTracker: modelRef.current?.memoryTracker ?? null,
    memorySummary,
  };
}
