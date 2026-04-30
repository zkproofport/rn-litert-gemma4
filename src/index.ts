import { Platform } from "react-native";
import type {
  Backend,
  GenerationStats,
  LLMConfig,
  LiteRTLM,
  MemoryUsage,
  Message,
  Role,
} from "./specs/LiteRTLM.nitro";

export type {
  Backend,
  GenerationStats,
  LLMConfig,
  LiteRTLM,
  MemoryUsage,
  Message,
  Role,
} from "./specs/LiteRTLM.nitro";

// Memory tracking utilities (uses NitroModules.createNativeArrayBuffer v0.35+).
export type {
  MemorySnapshot,
  MemoryTracker,
  MemoryTrackerSummary,
} from "./memoryTracker";
export { createMemoryTracker, createNativeBuffer } from "./memoryTracker";

export type { LiteRTLMInstance } from "./modelFactory";
export * from "./hooks";

/**
 * Creates a new LiteRT-LM inference engine instance.
 *
 * @example
 * ```typescript
 * import { createLLM, GEMMA_4_E2B_IT } from '@zkproofport/rn-litert-gemma4';
 *
 * const llm = createLLM();
 * await llm.loadModel('/path/to/gemma-4-E2B-it.litertlm', {
 *   backend: 'gpu',
 *   tools: JSON.stringify([
 *     {
 *       name: 'get_weather',
 *       description: 'Get current weather for a city',
 *       parameters: {
 *         type: 'object',
 *         properties: { city: { type: 'string' } },
 *         required: ['city'],
 *       },
 *     },
 *   ]),
 * });
 *
 * const raw = await llm.sendMessage('weather in Seoul?');
 * const call = parseGemma4ToolCall(raw);
 * if (call) {
 *   const result = await myToolRunner(call.name, call.arguments);
 *   // ...feed result back via sendMessage with a tool response.
 * }
 *
 * llm.close();
 * ```
 */
export { createLLM } from "./modelFactory";

/**
 * Pre-defined model identifiers — Gemma 4 only in this fork.
 * (Gemma 2/3, Phi, Qwen support was removed; see FORK.md.)
 */
export const Models = {
  /** Gemma 4 E2B Instruct (~2B effective params, 2.58 GB on disk) */
  GEMMA_4_E2B: "gemma-4-E2B-it-litert-lm",
  /** Gemma 4 E4B Instruct (~4B effective params, 3.65 GB on disk) */
  GEMMA_4_E4B: "gemma-4-E4B-it-litert-lm",
} as const;

export type ModelId = (typeof Models)[keyof typeof Models];

/**
 * Public download URL for Gemma 4 E2B IT (2.58 GB). No HuggingFace account
 * required — `litert-community` org publishes it openly.
 */
export const GEMMA_4_E2B_IT =
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm";

/**
 * Public download URL for Gemma 4 E4B IT (3.65 GB).
 */
export const GEMMA_4_E4B_IT =
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm";

/**
 * Get the recommended backend for the current platform.
 * GPU is preferred when available; falls back to CPU on unsupported devices.
 */
export function getRecommendedBackend(): Backend {
  return "gpu";
}

/**
 * Check if a backend configuration is supported. Returns a warning string if
 * the configuration may have issues, or undefined if OK.
 */
export function checkBackendSupport(backend: Backend): string | undefined {
  if (backend === "npu") {
    if (Platform.OS === "android") {
      return "NPU backend requires compatible hardware (Qualcomm Hexagon, MediaTek APU). Falls back to GPU if unavailable.";
    }
    if (Platform.OS === "ios") {
      return "NPU (Neural Engine) is not yet supported on iOS. Use 'gpu' (Metal) or 'cpu' instead.";
    }
  }
  return undefined;
}

/**
 * Multimodal vision/audio is supported on both Android and iOS in LiteRT-LM
 * v0.10.2+ (the engine version this fork ships against). The model file
 * (`gemma-4-E2B-it.litertlm`) bundles the vision encoder/adapter in a single
 * file — no `mmproj` side-load needed.
 */
export function checkMultimodalSupport(): string | undefined {
  return undefined;
}

/**
 * Result of parsing Gemma 4's native tool-call output.
 *
 * Gemma 4 emits tool calls as:
 *   `<|tool_call>call:NAME{key1:"value", key2:42}<tool_call|>`
 *
 * `arguments` is the raw JSON-ish object body. With
 * `enableConstrainedDecoding: true` on the engine, the body is guaranteed to
 * be parseable JSON; this helper attempts `JSON.parse` and returns the parsed
 * object, falling back to an empty object on failure (which only happens with
 * constrained decoding off + a model misbehavior).
 */
export interface Gemma4ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  /** The full matched substring, useful for stripping the call from `text`. */
  raw: string;
}

const GEMMA4_TOOL_CALL_RE =
  /<\|tool_call>\s*call:\s*([A-Za-z][\w-]*)\s*(\{[\s\S]*?\})\s*<tool_call\|>/;

/**
 * Extract a single tool call from a Gemma 4 model response.
 * Returns null if no tool call was emitted.
 */
export function parseGemma4ToolCall(text: string): Gemma4ToolCall | null {
  const match = GEMMA4_TOOL_CALL_RE.exec(text);
  if (!match) return null;
  const name = match[1];
  const body = match[2];
  if (!name || !body) return null;
  let args: Record<string, unknown> = {};
  try {
    // Gemma 4's tool call body is JSON-shaped (with constrained decoding).
    args = JSON.parse(body) as Record<string, unknown>;
  } catch {
    // Fallback: leave arguments empty rather than throwing.
  }
  return { name, arguments: args, raw: match[0] };
}

/**
 * Build a Gemma 4 tool-response block to send back to the model.
 *
 * Format (from Gemma 4 chat template):
 *   `<|tool_response>response:NAME{...result fields...}<tool_response|>`
 *
 * Pass the resulting string as the `message` argument to `llm.sendMessage`.
 */
export function buildGemma4ToolResponse(
  name: string,
  result: unknown,
): string {
  return `<|tool_response>response:${name}${JSON.stringify(result)}<tool_response|>`;
}
