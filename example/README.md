# example — RnMcpDemo

Test harness + interactive chat for `@zkproofport/rn-litert-gemma4`. Runs 13
automated scenarios (mobile actions, MCP round-trip, weather, direct FC) and
then hands a live LLM runtime to a chat panel for free-form testing.

## Setup

```bash
# from the fork root:
cd example
npm install               # links @zkproofport/rn-litert-gemma4 from `..`
cd ios && pod install && cd ..
```

iOS GPU acceleration requires the LiteRT-LM Metal/GPU dylibs to be embedded
in `App.app/Frameworks/`. The fork ships them under `../ios/dylibs/`. Until
podspec auto-embed (TODO), copy them manually after each build:

```bash
APP=ios/build-sim/Build/Products/Debug-iphonesimulator/RnMcpDemo.app
DYLIB=../ios/dylibs/ios_sim_arm64
for d in libLiteRt.dylib libLiteRtMetalAccelerator.dylib libGemmaModelConstraintProvider.dylib; do
  cp "$DYLIB/$d" "$APP/Frameworks/$d"
  codesign -fs - "$APP/Frameworks/$d"
done
codesign -fs - --deep "$APP"
```

## Models

The harness expects `gemma-4-E2B-it.litertlm` (or E4B) in the simulator's
Documents directory. The first launch shows a `ModelGate` UI that downloads
from HuggingFace if missing:

- E2B (~2.58 GB, default) — fast, lower quality
- E4B (~3.65 GB) — stronger but ~2-3x slower on CPU

## MCP server (for D1/D2 scenarios)

Start a local MCP server on the host:

```bash
npx -y supergateway \
  --stdio "npx -y @modelcontextprotocol/server-everything" \
  --port 4567 --outputTransport streamableHttp
```

The harness connects to `http://localhost:4567/mcp` with bearer token
`rn-mcp-demo`.

## Notes

- iOS Simulator falls back to CPU regardless of GPU dylib presence
  (LiteRT-LM Metal compute incompatible with the macOS GPU emulation
  layer). For real GPU benchmarks, build to a device.
- Speculative decoding is enabled unconditionally on engines that report
  drafter sections in the .litertlm bundle. Drafted/verified token counts
  stay 0 on CPU; effect only kicks in on device GPU.
