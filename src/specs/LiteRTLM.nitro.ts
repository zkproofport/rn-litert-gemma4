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
   * Primary compute backend for text generation.
   * - 'cpu': CPU inference (slower but always available)
   * - 'gpu': GPU acceleration (fast, recommended)
   * - 'npu': NPU/Neural Engine (fastest on supported devices)
   *
   * If not specified, defaults to 'gpu'.
   * If specified backend is unavailable, falls back automatically.
   *
   * @remarks
   * Vision encoder is always set to GPU (required by Gemma 3n).
   * Audio encoder is always set to CPU (optimal for audio processing).
   *
   * @default 'gpu'
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
 * LiteRT-LM: High-performance LLM inference engine.
 * Supports Gemma 3n, Phi-4, Qwen, and other .litertlm models.
 *
 * @example
 * ```typescript
 * const llm = createLLM();
 * llm.loadModel('/path/to/gemma-3n.litertlm', { backend: 'gpu' });
 *
 * // Blocking generation
 * const response = llm.sendMessage('What is the capital of France?');
 *
 * // Streaming generation
 * llm.sendMessageAsync('Tell me a story', (token, done) => {
 *   process.stdout.write(token);
 * });
 *
 * llm.close();
 * ```
 */
export interface LiteRTLM extends HybridObject<{
  ios: "c++";
  android: "kotlin";
}> {
  /**
   * Load a .litertlm model file.
   * @param config Optional configuration for backend and sampling.
   * @throws Error if the model cannot be loaded.
   */
  loadModel(modelPath: string, config?: LLMConfig): Promise<void>;

  /**
   * Send a text message and get the complete response (blocking).
   * @param message User message text.
   * @returns The model's response text.
   */
  sendMessage(message: string): Promise<string>;

  /**
   * Send a text message with an image (multimodal).
   * @param message User message text.
   * @param imagePath Absolute path to an image file.
   * @returns The model's response text.
   */
  sendMessageWithImage(message: string, imagePath: string): Promise<string>;

  /**
   * Send a text message with audio (multimodal).
   * @param message User message text.
   * @param audioPath Absolute path to an audio file (WAV).
   * @returns The model's response text.
   */
  sendMessageWithAudio(message: string, audioPath: string): Promise<string>;

  /**
   * Send a message with streaming response.
   * Tokens are delivered via callback as they are generated.
   * @param message User message text.
   * @param onToken Callback invoked for each token (token, isDone).
   */
  sendMessageAsync(
    message: string,
    onToken: (token: string, done: boolean) => void,
  ): void;

  /**
   * Get the current conversation history.
   * @returns Array of messages in the conversation.
   */
  getHistory(): Message[];

  /**
   * Clear the conversation context and start fresh.
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
   * Release all native resources.
   * Call this when done with the LLM instance.
   */
  close(): void;
}
