# Fork rationale

`@zkproofport/rn-litert-gemma4` is a hard fork of [`react-native-litert-lm` v0.3.6](https://github.com/hung-yueh/react-native-litert-lm) by Hugh Chen, taken on 2026-04-30 against commit `92a6427`.

We forked instead of patching because the upstream JS layer is several model generations behind the LiteRT-LM C engine. Trying to PR our changes upstream would block downstream work for weeks; owning the JS surface lets us move at our own pace and ship a Gemma 4-first API.

## What stays from upstream

- `cpp/HybridLiteRTLM.{cpp,hpp}` — the Nitro-based C++ wrapper that talks to the LiteRT-LM C engine. We still call its functions, but extend the surface.
- `cpp/include/litert_lm_engine.h` — Google's LiteRT-LM C API header (vendored). All function-calling capability comes from here.
- iOS `LiteRTLMAutolinking.mm` removed in 0.3.6, kept removed.
- Android Kotlin bridge — kept as-is.
- Nitrogen code generation — kept; we regenerate after spec changes.
- iOS XCFramework download/build scripts — kept.

## What is removed (Gemma 4-first focus)

- `src/templates.ts`, `lib/templates.js`. The upstream `applyGemmaTemplate`, `applyPhiTemplate`, `applyLlamaTemplate` helpers all emit chat templates from older model generations (Gemma 2/3, Phi, Llama 3) that drift from Gemma 4's actual format. They are replaced by direct use of the engine's `litert_lm_conversation_config_create` which takes a structured `tools_json` and lets the engine emit the correct template internally.
- Phi / Llama support. Out of scope for this fork. Use upstream `react-native-litert-lm` if you need them.
- `release-it` and the upstream release pipeline. Internal package.

## What is added

- `tools_json` parameter on `loadModel` (or a new `setTools` call) so callers pass an MCP/JSON-Schema-style tool list.
- `enable_constrained_decoding=true` by default — the engine forces Gemma 4's six special tool tokens (`<|tool_call>`, `<tool_call|>`, `<|tool_response>`, `<tool_response|>`, `<|tool>`, `<tool|>`), so the JS side never regex-parses raw text.
- Structured `ToolCall` return type from `sendMessage*` — the wrapper extracts `name` + `args` from the engine output and hands them back as objects.
- Multimodal-aware tool calls (`sendMessageWithImage` / `sendMessageWithAudio` continue working; tools registered with `loadModel` apply across all input modalities).

## What is intentionally left for later

- Gemma 5 support. Will be evaluated when released.
- Streaming `ToolCall` deltas. The engine supports it; we will expose once the first synchronous version is solid.
- Schema validation of `tools_json` on the JS side. The engine will reject bad JSON; we will mirror that to a TypeScript type later.

## Upstream tracking

`upstream` remote points at `hung-yueh/react-native-litert-lm`. We will pull engine version bumps but keep the JS surface in our hands.
