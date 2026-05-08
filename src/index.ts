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

// Fallback: Gemma 4 (without constrained decoding) often emits a ```json
// fenced code block instead of native <|tool_call|> tokens. Detect both
// `{ "tool_name": "X", "parameters": {...} }` and `{ "name": "X",
// "arguments": {...} }` (OpenAI-style) shapes.
//
// NOTE: Implemented with indexOf+slice instead of a regex because Hermes JS
// engine (default React Native runtime) compiles regex literals to bytecode
// that misbehaves on multi-line dotall captures with backtick fences.
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

/**
 * Strip Gemma 4 / FunctionGemma string-quoting tokenizer artifacts.
 *
 * The model wraps literal-string arguments with `<|"|>...<|"|>` markers when
 * emitting tool calls. The engine's chat-template decoder normally consumes
 * these, but they leak through when constrained decoding is OFF or the
 * structured tool_calls pathway is taken. We strip them here so callers see
 * clean values.
 */
function stripStringArtifacts(value: unknown): unknown {
  if (typeof value === 'string') {
    let v = value;
    // Repeatedly strip <|"|>...<|"|> wrapping.
    let prev: string;
    do {
      prev = v;
      v = v.replace(/^<\|"\|>/, '').replace(/<\|"\|>$/, '');
    } while (v !== prev);
    return v.trim();
  }
  if (Array.isArray(value)) return value.map(stripStringArtifacts);
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = stripStringArtifacts(v);
    }
    return out;
  }
  return value;
}

/**
 * Extract a single tool call from a Gemma 4 model response.
 * Returns null if no tool call was emitted.
 *
 * Recognizes three output formats:
 *   1. Structured: `{"role":"assistant","tool_calls":[{"type":"function","function":{"name":"X","arguments":{...}}}]}`
 *      — emitted by the runtime's `Conversation::SendMessage` pipeline once
 *      the engine has parsed native tool tokens. This is the primary path
 *      with the C++-direct iOS wrapper.
 *   2. Native (legacy): `<|tool_call>call:NAME{...}<tool_call|>`
 *   3. JSON fence (legacy): ` ```json\n{ "tool_name": "X", "parameters": {...} }\n``` `
 */
export function parseGemma4ToolCall(text: string): Gemma4ToolCall | null {
  // 1. Structured Message JSON from the runtime's tool_calls pipeline.
  // Detect heuristically — full JSON parse can fail on partial/oddly-quoted
  // chunks; require the literal field marker.
  if (text.includes('"tool_calls"')) {
    try {
      const obj = JSON.parse(text) as Record<string, unknown>;
      const calls = obj.tool_calls as
        | Array<{ function?: { name?: string; arguments?: unknown } }>
        | undefined;
      const first = calls && calls[0];
      const fn = first && first.function;
      if (fn && typeof fn.name === 'string' && fn.name.length > 0) {
        const argsClean = stripStringArtifacts(fn.arguments ?? {}) as Record<
          string,
          unknown
        >;
        return { name: fn.name, arguments: argsClean, raw: text };
      }
    } catch {}
  }

  const native = GEMMA4_TOOL_CALL_RE.exec(text);
  if (native) {
    const name = native[1];
    const body = native[2];
    if (name && body) {
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(body) as Record<string, unknown>;
      } catch {}
      return {
        name,
        arguments: stripStringArtifacts(args) as Record<string, unknown>,
        raw: native[0],
      };
    }
  }

  const body = extractJsonFenceBody(text);
  if (body) {
    try {
      const obj = JSON.parse(body) as Record<string, unknown>;
      const name =
        (obj.tool_name as string | undefined) ?? (obj.name as string | undefined);
      const argsObj =
        (obj.parameters as Record<string, unknown> | undefined) ??
        (obj.arguments as Record<string, unknown> | undefined) ??
        {};
      if (typeof name === 'string' && name.length > 0) {
        return {
          name,
          arguments: stripStringArtifacts(argsObj) as Record<string, unknown>,
          raw: body,
        };
      }
    } catch {}
  }

  return null;
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
