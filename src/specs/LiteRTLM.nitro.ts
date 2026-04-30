import type { HybridObject } from "react-native-nitro-modules";

/**
 * Backend types for LLM inference.
 * - 'cpu': CPU inference (slowest, always available)
 * - 'gpu': GPU acceleration (fast, recommended for most devices)
 * - 'npu': NPU/Neural Engine (fastest on supported hardware)
 *
 * @remarks
 * NPU acceleration requires compatible hardware (e.g., Qualcomm Hexagon, MediaTek APU).
 * If NPU is unavailable, LiteRT-LM automatically falls back to GPU.
 */
export type Backend = "cpu" | "gpu" | "npu";

/**
 * Message roles for conversation.
 */
export type Role = "user" | "model" | "system";

/**
 * Configuration options for loading an LLM.
 */
export interface LLMConfig {
  /**
   * System prompt to set the model's behavior.
   * This is prepended to the conversation to guide model responses.
   * @example "You are a helpful coding assistant."
   */
  systemPrompt?: string;

  /**
   * Primary compute backend for text generation.
   * - 'cpu': CPU inference (safe default, always available)
   * - 'gpu': GPU acceleration (fast, Metal on iOS, GPU delegate on Android)
   * - 'npu': NPU/Neural Engine (fastest on supported devices)
   *
   * If not specified, defaults to 'cpu'.
   * If specified backend is unavailable, falls back automatically.
   *
   * @default 'cpu'
   */
  backend?: Backend;

  /**
   * Maximum number of tokens to generate.
   * @default 1024
   */
  maxTokens?: number;

  /**
   * Sampling temperature (0.0 = deterministic, 1.0 = creative).
   * @default 0.7
   */
  temperature?: number;

  /**
   * Top-K sampling (number of top tokens to consider).
   * @default 40
   */
  topK?: number;

  /**
   * Top-P (nucleus) sampling threshold.
   * @default 0.95
   */
  topP?: number;

  /**
   * Tool definitions in Gemma 4-compatible JSON array format.
   *
   * When set, the LiteRT-LM engine builds the proper
   * `<|tool>declaration:NAME{...}<tool|>` block in the chat template and the
   * model emits `<|tool_call>call:NAME{...}<tool_call|>` when it decides to
   * call a tool. With `enableConstrainedDecoding: true` (the default when
   * `tools` is set), the engine forces well-formed tool calls at the token
   * level — no regex parsing needed.
   *
   * Example:
   * ```json
   * [
   *   {
   *     "name": "get_weather",
   *     "description": "Get current temperature for a city",
   *     "parameters": {
   *       "type": "object",
   *       "properties": { "city": { "type": "string" } },
   *       "required": ["city"]
   *     }
   *   }
   * ]
   * ```
   */
  tools?: string;

  /**
   * Force the model to emit only well-formed tool calls / tool responses
   * via constrained decoding at the token level. Defaults to `true` when
   * `tools` is non-empty, `false` otherwise.
   */
  enableConstrainedDecoding?: boolean;
}

/**
 * A simple message in the conversation.
 * For multimodal, use sendMessageWithImage/sendMessageWithAudio instead.
 */
export interface Message {
  /** Role of the message sender */
  role: Role;
  /** Text content of the message */
  content: string;
}

/**
 * Generation statistics returned after completion.
 */
export interface GenerationStats {
  /** Number of tokens in the prompt */
  promptTokens: number;
  /** Number of tokens generated */
  completionTokens: number;
  /** Total tokens (prompt + completion) */
  totalTokens: number;
  /** Time to first token in milliseconds */
  timeToFirstToken: number;
  /** Total generation time in milliseconds */
  totalTime: number;
  /** Tokens per second */
  tokensPerSecond: number;
}

/**
 * Real memory usage statistics from the native runtime.
 * Measured from OS-level APIs, not estimated.
 */
export interface MemoryUsage {
  /** Native heap allocated bytes */
  nativeHeapBytes: number;
  /** Total process resident set size (RSS) in bytes */
  residentBytes: number;
  /** Available system memory in bytes */
  availableMemoryBytes: number;
  /** Whether the system considers memory low */
  isLowMemory: boolean;
}

/**
 * LiteRT-LM: High-performance Gemma 4 inference engine.
 *
 * @example
 * ```typescript
 * const llm = createLLM();
 * await llm.loadModel('/path/to/gemma-4-E2B-it.litertlm', {
 *   backend: 'gpu',
 *   tools: JSON.stringify([{ name: 'get_weather', ... }]),
 * });
 * const text = await llm.sendMessage('weather in Seoul?');
 * llm.close();
 * ```
 */
export interface LiteRTLM extends HybridObject<{
  ios: "c++";
  android: "kotlin";
}> {
  /**
   * Load a .litertlm model file.
   * @param config Optional configuration for backend, sampling, and tools.
   */
  loadModel(modelPath: string, config?: LLMConfig): Promise<void>;

  /**
   * Replace the active tools list without reloading the model.
   * Pass an empty string to clear tools.
   *
   * Internally calls `litert_lm_conversation_config_create` with the new
   * tools JSON; constrained decoding is auto-enabled when tools is non-empty.
   */
  setTools(toolsJson: string): Promise<void>;

  /**
   * Send a text message and get the complete response (blocking).
   * @returns Raw model output. With `tools` set, this includes Gemma 4's
   * native `<|tool_call>...<tool_call|>` tokens — use the `parseGemma4ToolCall`
   * helper from this package's JS layer to extract a structured ToolCall.
   */
  sendMessage(message: string): Promise<string>;

  /**
   * Send a text message with an image (multimodal).
   */
  sendMessageWithImage(message: string, imagePath: string): Promise<string>;

  /**
   * Download a model file from a URL.
   */
  downloadModel(
    url: string,
    fileName: string,
    onProgress?: (progress: number) => void,
  ): Promise<string>;

  /**
   * Delete a downloaded model file.
   */
  deleteModel(fileName: string): Promise<void>;

  /**
   * Send a text message with audio (multimodal).
   */
  sendMessageWithAudio(message: string, audioPath: string): Promise<string>;

  /**
   * Send a message with streaming response.
   */
  sendMessageAsync(
    message: string,
    onToken: (token: string, done: boolean) => void,
  ): void;

  /**
   * Get the current conversation history.
   */
  getHistory(): Message[];

  /**
   * Clear the conversation context and start fresh. Tools are preserved.
   */
  resetConversation(): void;

  /**
   * Check if a model is loaded and ready for inference.
   */
  isReady(): boolean;

  /**
   * Get the last generation statistics.
   */
  getStats(): GenerationStats;

  /**
   * Get real memory usage from the native runtime.
   */
  getMemoryUsage(): MemoryUsage;

  /**
   * Release all native resources.
   */
  close(): void;
}
