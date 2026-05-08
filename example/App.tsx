/**
 * RnMcpDemo — Step D: native Gemma 4 FC with REAL FST (constrained decoding ON).
 *
 * Resolution: stubbed `gemma_model_constraint_provider.o` was removed from
 * LiteRTLM.xcframework's static archive, and the prebuilt
 * libGemmaModelConstraintProvider.dylib (from LiteRT-LM v0.10.2's
 * prebuilt/ios_arm64 + prebuilt/ios_sim_arm64 tree, shipped by Google with
 * LLguidance statically linked)
 * is now embedded as a separate XCFramework. This restores real FST-based
 * constrained decoding so the model is forced to emit
 * `<|tool_call>call:NAME{...}<tool_call|>` tokens.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  Linking,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  useColorScheme,
} from 'react-native';
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';
import RNFS from 'react-native-fs';

import {
  buildGemma4ToolResponse,
  createLLM,
  parseGemma4ToolCall,
  type Gemma4ToolCall,
} from '@zkproofport/rn-litert-gemma4';
import { McpHttpClient } from '@zkproofport/rn-on-device-mcp';

// =============================================================================
// Model catalog — pick chat model at runtime, download on demand.
// =============================================================================
//
// Mirrors Edge Gallery upstream `model_allowlists/1_0_13.json` for the
// `llm_agent_commerce_chat` task — only Gemma 4 E2B and E4B are listed there.
// FunctionGemma 270M is intentionally NOT included (Edge Gallery upstream does
// not use it for agent chat; we previously experimented with it as a "direct
// FC baseline" but that path is gone now — single model + load_skill MUST
// pattern is the only supported flow).
//
// Each entry exposes:
//   - filename: must match what we look for under Documents/[models/]
//   - url: HuggingFace litert-community URL (open access, no auth)
//   - sizeBytes: used for the download progress bar / "still need to fetch" UI
interface ModelEntry {
  key: string;
  label: string;
  filename: string;
  url: string;
  sizeBytes: number;
}

const MODEL_CATALOG: ModelEntry[] = [
  {
    key: 'gemma-4-E2B',
    label: 'Gemma 4 E2B (2.58 GB)',
    filename: 'gemma-4-E2B-it.litertlm',
    url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
    sizeBytes: 2_588_147_712, // exact size from Edge Gallery model_allowlists
  },
  {
    key: 'gemma-4-E4B',
    label: 'Gemma 4 E4B (3.65 GB — stronger but slower)',
    filename: 'gemma-4-E4B-it.litertlm',
    url: 'https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm',
    sizeBytes: 3_659_530_240, // exact size from Edge Gallery model_allowlists
  },
];

const DEFAULT_CHAT_KEY = 'gemma-4-E2B';
const RESULTS_FILE = 'test-results.json';

const findModelPath = async (filename: string): Promise<string> => {
  const cand = [
    `${RNFS.DocumentDirectoryPath}/models/${filename}`,
    `${RNFS.DocumentDirectoryPath}/${filename}`,
  ];
  for (const c of cand) {
    if (await RNFS.exists(c)) return c;
  }
  return '';
};

interface DownloadState {
  inProgress: boolean;
  bytesWritten: number;
  totalBytes: number;
  jobId?: number;
  error?: string;
}

const initialDownload: DownloadState = {
  inProgress: false,
  bytesWritten: 0,
  totalBytes: 0,
};

const formatMB = (bytes: number): string =>
  `${(bytes / 1024 / 1024).toFixed(1)} MB`;

// Read the LiteRT-LM stderr log and extract the most recent
// `[FORK] backend_resolved=...` line. Returns 'GPU' / 'CPU' / 'NPU' or
// 'unknown' if the marker isn't found yet (e.g. before first loadModel).
//
// Why JS-side parsing of the log file: the native LLMConfig only carries
// the *requested* backend; the actual resolved backend (after fallback
// chain) is logged to the stderr file by HybridLiteRTLM but isn't exposed
// through the Nitro API yet. Reading the log avoids a native rebuild.
async function readResolvedBackend(): Promise<'GPU' | 'CPU' | 'NPU' | 'unknown'> {
  const path = `${RNFS.DocumentDirectoryPath}/litert-stderr.log`;
  try {
    if (!(await RNFS.exists(path))) return 'unknown';
    const log = await RNFS.readFile(path, 'utf8');
    // Find LAST occurrence so a model switch reflects the latest result.
    const matches = [...log.matchAll(/\[FORK\] backend_resolved=(GPU|CPU|NPU)/g)];
    if (matches.length === 0) return 'unknown';
    const last = matches[matches.length - 1];
    return last && last[1] ? (last[1] as 'GPU' | 'CPU' | 'NPU') : 'unknown';
  } catch {
    return 'unknown';
  }
}

// MCP test endpoint — Anthropic's `@modelcontextprotocol/server-everything`
// reference server fronted by `supergateway` for streamable-HTTP transport.
// Spin it up on the host with:
//   npx -y supergateway --stdio "npx -y @modelcontextprotocol/server-everything" --port 4567 --outputTransport streamableHttp
// iOS simulator reaches host services at `localhost`. The server accepts any
// non-empty bearer token; we use a fixed dev token for traceability.
const MCP_URL = 'http://localhost:4567/mcp';
const MCP_AUTH_TOKEN = 'rn-mcp-demo';
const mcpClient = new McpHttpClient({
  url: MCP_URL,
  authToken: MCP_AUTH_TOKEN,
  clientInfo: { name: 'RnMcpDemo', version: '0.1.0' },
});

// =============================================================================
// Skill Registry — runtime-extensible, model-routable via load_skill
// =============================================================================
//
// Architecture mimics Google AI Edge Gallery's load_skill / run_js / run_intent
// pattern (https://github.com/google-ai-edge/gallery/tree/main/Android/src/app/src/main/assets/skills).
// The model NEVER decides "should I call a tool" — the system prompt mandates
// load_skill first when a skill matches; load_skill returns instructions that
// tell the model exactly which run_* tool to call next.

interface SkillDef {
  name: string;
  description: string;
  instructions: string;
}

const SKILLS: SkillDef[] = [
  {
    name: 'general_chat',
    description: 'For casual conversation, greetings, opinions, or general knowledge questions you can answer from your own training (e.g., "what is the capital of France", "how are you", fun facts).',
    instructions:
      'Respond to the user directly in 1-3 sentences from your own knowledge. Do NOT call any other tool. If you do not know the answer, plainly say "I do not know" and propose ONE specific follow-up question they could ask.',
  },
  {
    name: 'weather',
    description: 'Get current weather (temperature, conditions, precipitation, wind) or a 7-day forecast for a specific city. Requires a city name; if the user mentions only a duration ("this week", "tomorrow") without a city, ask which city.',
    instructions:
      'PREREQUISITE: parse a city name from the user message (English, Korean, or any language). If NO city is mentioned AND the conversation has no prior city context, reply: "Which city would you like the weather for?" and stop. Otherwise reuse the last-mentioned city from prior turns.\n' +
      'Detect intent: "current/now/right now/지금" → use CURRENT path. "this week/forecast/tomorrow/이번주/내일" → use FORECAST path.\n' +
      '\n' +
      'COMMON Step 1 (geocoding): Call run_api with method="GET" and url="https://geocoding-api.open-meteo.com/v1/search?name=<URL-encoded English city name>&count=1". Response: results[0].latitude, results[0].longitude. If empty, reply: "I could not find a city named <city>." and stop.\n' +
      '\n' +
      'CURRENT path Step 2: Call run_api with method="GET" and url="https://api.open-meteo.com/v1/forecast?latitude=<lat>&longitude=<lon>&current=temperature_2m,weather_code,wind_speed_10m,precipitation". The response includes current.temperature_2m (°C), current.weather_code (WMO), current.wind_speed_10m (km/h), current.precipitation (mm).\n' +
      'CURRENT Step 3: Translate weather_code per the WMO TABLE below, then reply with one or two sentences. Example: "Seoul is 14°C, partly cloudy, with light wind (3 km/h)."\n' +
      '\n' +
      'FORECAST path Step 2: Call run_api with method="GET" and url="https://api.open-meteo.com/v1/forecast?latitude=<lat>&longitude=<lon>&daily=temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum&timezone=auto&forecast_days=7". The response includes daily.time[], daily.temperature_2m_max[], daily.temperature_2m_min[], daily.weather_code[], daily.precipitation_sum[].\n' +
      'FORECAST Step 3: Summarize the 7 days in one paragraph (max 6 sentences). Mention range (min/max), dominant condition (translated weather_code), and any rainy/snowy days. Example: "Seoul this week: highs 12–18°C, lows 6–10°C; mostly cloudy with rain on Wed and Sat."\n' +
      '\n' +
      'WMO weather_code TABLE (memorize these mappings — translate to natural English/Korean as appropriate):\n' +
      '  0 = clear sky\n' +
      '  1 = mainly clear, 2 = partly cloudy, 3 = overcast\n' +
      '  45 = fog, 48 = depositing rime fog\n' +
      '  51 = light drizzle, 53 = moderate drizzle, 55 = dense drizzle\n' +
      '  61 = slight rain, 63 = moderate rain, 65 = heavy rain\n' +
      '  71 = slight snow, 73 = moderate snow, 75 = heavy snow\n' +
      '  77 = snow grains\n' +
      '  80 = slight rain showers, 81 = moderate rain showers, 82 = violent rain showers\n' +
      '  85 = slight snow showers, 86 = heavy snow showers\n' +
      '  95 = thunderstorm, 96 = thunderstorm with slight hail, 99 = thunderstorm with heavy hail',
  },
  {
    name: 'wikipedia',
    description: 'Look up a Wikipedia summary for a specific topic/person/place/event. Requires a clear topic name.',
    instructions:
      'PREREQUISITE: identify a single specific topic from the user message. If the topic is unclear or missing, do NOT call any tool — reply: "Which topic would you like a Wikipedia summary for?" and stop.\n' +
      'If a topic IS specified:\n' +
      'Step 1: Call run_api with method="GET" and url="https://en.wikipedia.org/api/rest_v1/page/summary/<URL-encoded topic>". The response field "extract" contains the summary.\n' +
      'Step 2: Reply with the extract trimmed to 2–3 sentences. If the response indicates "not found", reply: "I could not find a Wikipedia article for <topic>." and suggest 1 alternative search term.',
  },
  {
    name: 'add_numbers',
    description: 'Compute the exact sum of two specific integers. Both numbers must come from the user message.',
    instructions:
      'PREREQUISITE: parse two integers from the user message. If you cannot find two numbers, do NOT call any tool — reply: "Please tell me which two numbers to add." and stop.\n' +
      'If two numbers are present:\n' +
      'Call run_api with method="POST", url="data:application/json,sum", body={"a": <a>, "b": <b>}. The handler returns the exact sum.\n' +
      'Reply with one sentence: "The sum of <a> and <b> is <sum>."',
  },
  {
    name: 'echo',
    description: 'Repeat the user-supplied text verbatim.',
    instructions:
      'PREREQUISITE: identify the exact text the user wants repeated. If unclear, reply: "What would you like me to repeat?" and stop.\n' +
      'Otherwise reply with EXACTLY that text — no extra words, no quotes, no commentary.',
  },
  // ── Mobile action skills (Edge Gallery MobileActions parity, but exposed as
  //    skills so load_skill mediation bypasses RLHF refusal) ─────────────────
  {
    name: 'send_email',
    description: 'Open the iOS email composer pre-filled with recipient/subject/body. Requires at least a recipient email address.',
    instructions:
      'PREREQUISITE: parse a recipient email from the user message. If absent, reply: "Who should I send the email to?" and stop.\n' +
      'If recipient is present:\n' +
      'Call run_intent with intent="send_email" and parameters set to a JSON string, e.g. parameters="{\\"to\\":\\"bob@example.com\\",\\"subject\\":\\"Meeting\\",\\"body\\":\\"3pm\\"}". Replace values with the user-provided ones. If subject is missing, use a one-line summary of the body. If body is missing, leave it empty.\n' +
      'The handler returns a status field. If status="opened mail composer", reply: "Opened the email composer for <to>." If status indicates failure, reply with the error message verbatim.',
  },
  {
    name: 'set_alarm',
    description: 'Try to set an alarm at a specific time. iOS does not expose a public alarm API — this is expected to return "unsupported".',
    instructions:
      'PREREQUISITE: parse a time (HH:MM 24-hour) from the user message. If absent, reply: "What time should I set the alarm for?" and stop.\n' +
      'Otherwise call run_intent with intent="set_alarm" and parameters="{\\"time\\":\\"<HH:MM>\\",\\"label\\":\\"<label>\\"}". Convert 12-hour times to 24-hour. The handler will return status="unsupported" because iOS has no public alarm API. Reply: "I cannot set alarms on iOS; please open the Clock app manually."',
  },
  {
    name: 'open_app',
    description: 'Open a built-in iOS app via its public URL scheme. Supported app names: calendar, maps, settings, mail, messages, facetime, phone, reminders.',
    instructions:
      'PREREQUISITE: parse an app name from the user message. If absent or unrecognizable, reply: "Which app should I open?" and stop.\n' +
      'Otherwise call run_intent with intent="open_app" and parameters="{\\"appName\\":\\"<name in lowercase>\\"}". The handler returns a status field. If status="opened", reply: "Opening <appName>." If status="unsupported", reply with the exact error message — DO NOT pretend the app opened.',
  },
  // ── MCP-backed skill (real protocol round-trip, not a stub) ──────────────
  {
    name: 'mcp_sum',
    description: 'Add two integers exactly via the MCP server\'s get-sum tool. Both numbers must be parsed from the user message.',
    instructions:
      'PREREQUISITE: parse two integers from the user message. If you cannot find two numbers, reply: "Please tell me which two numbers to add." and stop.\n' +
      'Otherwise call run_mcp_tool with serverName="everything", toolName="get-sum", and parameters="{\\"a\\":<a>,\\"b\\":<b>}" (replace <a> and <b> with the integers). The response includes result. Reply with one sentence: "MCP says <a> + <b> = <sum>."',
  },
  {
    name: 'mcp_echo',
    description: 'Echo a message through the MCP server\'s echo tool. Requires the message text from the user.',
    instructions:
      'PREREQUISITE: identify the exact text the user wants echoed. If unclear, reply: "What text should I echo through MCP?" and stop.\n' +
      'Otherwise call run_mcp_tool with serverName="everything", toolName="echo", parameters="{\\"message\\":\\"<text>\\"}" (replace <text> with the actual message). The response includes the echoed text. Reply: "MCP echoed: <echoed>."',
  },
];

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'load_skill',
      description:
        'Load a skill by name. Returns step-by-step instructions describing exactly which subsequent tools to call (run_api or run_mcp_tool). You MUST call this BEFORE responding whenever the user request matches one of the available skills listed in the system prompt. Do NOT reply to the user before calling load_skill if a skill applies.',
      parameters: {
        type: 'object',
        properties: {
          skillName: { type: 'string', description: 'The name of the skill to load (must match one in the available skills list).' },
        },
        required: ['skillName'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_api',
      description:
        'Make an HTTP API call. Used by skills that need to fetch external data. Returns the raw response body parsed as JSON when possible, otherwise as text.',
      parameters: {
        type: 'object',
        properties: {
          method: { type: 'string', description: 'HTTP method: GET, POST, PUT, DELETE.' },
          url: { type: 'string', description: 'The full URL to request.' },
          headers: { type: 'object', description: 'Optional HTTP headers as key/value pairs.' },
          body: { type: 'object', description: 'Optional JSON body for POST/PUT requests.' },
        },
        required: ['method', 'url'],
      },
    },
  },
  {
    // Same flat-strings-only pattern as run_intent (mirrors Edge Gallery's
    // AgentTools.runJs/runIntent: nested objects → JSON strings).
    type: 'function',
    function: {
      name: 'run_mcp_tool',
      description: 'Call a tool exposed by a connected MCP (Model Context Protocol) server.',
      parameters: {
        type: 'object',
        properties: {
          serverName: { type: 'string', description: 'The MCP server identifier.' },
          toolName: { type: 'string', description: 'The MCP tool name within that server.' },
          parameters: {
            type: 'string',
            description: 'A JSON string containing the parameter values required for the MCP tool.',
          },
        },
        required: ['serverName', 'toolName', 'parameters'],
      },
    },
  },
  {
    // Mirrors Edge Gallery's AgentTools.runIntent(intent: String, parameters: String):
    //   @ToolParam("The intent to run.") intent
    //   @ToolParam("A JSON string containing the parameter values required for
    //              the intent.") parameters
    // The Kotlin SDK auto-derives the schema from @Tool/@ToolParam reflection;
    // both fields surface as plain `{type: "string"}`. We replicate the same
    // OpenAI-style schema by hand here.
    type: 'function',
    function: {
      name: 'run_intent',
      description: 'Run an intent. It is used to interact with the app to perform certain actions.',
      parameters: {
        type: 'object',
        properties: {
          intent: { type: 'string', description: 'The intent to run.' },
          parameters: {
            type: 'string',
            description: 'A JSON string containing the parameter values required for the intent.',
          },
        },
        required: ['intent', 'parameters'],
      },
    },
  },
];

// DIRECT_TOOLS — flat-scalar tool definitions for the "direct FC" path.
// The same Gemma 4 model is loaded a second time with a minimal system prompt
// (no load_skill mandate) and these tools. Comparing direct-FC scenarios to
// the load_skill-mediated ones validates that the model can call tools
// without skill indirection — at the cost of being more vulnerable to RLHF
// refusal on action-style requests.
//
// All param slots are flat scalars (string/number) because constrained
// decoding emits nested objects unreliably. MCP tools that need structured
// args go through dedicated wrappers (mcp_sum/mcp_echo) instead of the
// generic run_mcp_tool envelope.
const DIRECT_TOOLS = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current temperature in a city. Returns Celsius.',
      parameters: {
        type: 'object',
        properties: {
          city: { type: 'string', description: 'City name in English, e.g. Seoul, Tokyo, Madrid.' },
        },
        required: ['city'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'mcp_sum',
      description: 'Add two integers exactly via the MCP get-sum tool. Use this when the user asks to sum specific numbers and accuracy matters.',
      parameters: {
        type: 'object',
        properties: {
          a: { type: 'number', description: 'First integer.' },
          b: { type: 'number', description: 'Second integer.' },
        },
        required: ['a', 'b'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'mcp_echo',
      description: 'Echo a message through the MCP echo tool to verify the round-trip is alive.',
      parameters: {
        type: 'object',
        properties: {
          message: { type: 'string', description: 'Message to echo.' },
        },
        required: ['message'],
      },
    },
  },
];

async function executeTool(call: Gemma4ToolCall): Promise<unknown> {
  const args = call.arguments;
  // Direct tool handlers (used by DIRECT_TOOLS path)
  if (call.name === 'get_weather') {
    const city = (args.city as string) ?? (args.location as string) ?? '';
    const geo = await fetch(
      `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(city)}&count=1`,
    ).then((r) => r.json() as Promise<{ results?: { latitude: number; longitude: number }[] }>);
    const hit = geo.results?.[0];
    if (!hit) return { error: 'city not found' };
    const wx = await fetch(
      `https://api.open-meteo.com/v1/forecast?latitude=${hit.latitude}&longitude=${hit.longitude}&current=temperature_2m`,
    ).then((r) => r.json() as Promise<{ current?: { temperature_2m: number } }>);
    return { city, temperatureC: wx.current?.temperature_2m };
  }
  if (call.name === 'add_numbers') {
    const a = Number(args.a);
    const b = Number(args.b);
    return { result: a + b };
  }
  if (call.name === 'echo') {
    return { echoed: (args.text as string) ?? '' };
  }
  if (call.name === 'load_skill') {
    const name = (args.skillName as string) ?? '';
    const skill = SKILLS.find((s) => s.name === name);
    if (!skill) return { error: `unknown skill: ${name}` };
    return { skill_name: skill.name, instructions: skill.instructions };
  }
  if (call.name === 'run_api') {
    const method = ((args.method as string) ?? 'GET').toUpperCase();
    const url = (args.url as string) ?? '';
    const headers = (args.headers as Record<string, string>) ?? {};
    const body = args.body as Record<string, unknown> | undefined;
    // Special case: data:application/json,sum — local handler for add_numbers skill demo
    if (url.startsWith('data:application/json,sum')) {
      const a = Number((body as Record<string, unknown> | undefined)?.a ?? 0);
      const b = Number((body as Record<string, unknown> | undefined)?.b ?? 0);
      return { result: a + b };
    }
    try {
      const res = await fetch(url, {
        method,
        headers: { ...headers, ...(body ? { 'Content-Type': 'application/json' } : {}) },
        body: body ? JSON.stringify(body) : undefined,
      });
      const text = await res.text();
      try {
        return { status: res.status, body: JSON.parse(text) };
      } catch {
        return { status: res.status, body: text };
      }
    } catch (err) {
      return { error: String(err) };
    }
  }
  if (call.name === 'run_mcp_tool') {
    // Real MCP tool call via streamable-HTTP transport. Connects to the
    // server identified by `serverName` (currently we expose just one,
    // "everything", pinned to MCP_URL). The MCP server-everything reference
    // is running locally; we forward toolName/params verbatim.
    const serverName = (args.serverName as string) ?? '';
    const toolName = (args.toolName as string) ?? '';
    // `parameters` is a JSON-encoded string (matches Kotlin's
    // AgentTools.runIntent + runJs convention). Constrained decoding
    // can only reliably emit flat slots, so nested values flow as JSON.
    let params: Record<string, unknown> = {};
    const rawParams = args.parameters ?? args.params;
    if (typeof rawParams === 'string' && rawParams.trim()) {
      try {
        const parsed = JSON.parse(rawParams);
        if (parsed && typeof parsed === 'object') params = parsed as Record<string, unknown>;
      } catch (err) {
        return { error: `parameters JSON parse failed: ${String(err)}`, parameters: rawParams };
      }
    } else if (rawParams && typeof rawParams === 'object') {
      params = rawParams as Record<string, unknown>;
    }
    if (serverName !== 'everything') {
      return { error: `unknown MCP server: ${serverName}`, expected: 'everything' };
    }
    if (!toolName) {
      return { error: 'toolName required' };
    }
    try {
      const raw = await mcpClient.callTool(toolName, params);
      // Unwrap the MCP {content:[{type:'text',text:'...'}]} envelope so the
      // model sees a clean structured result. Fall back to raw if shape is
      // unexpected.
      const env = raw as { content?: Array<{ type?: string; text?: string }> } | undefined;
      const first = env?.content?.[0];
      if (first?.type === 'text' && typeof first.text === 'string') {
        try {
          return { result: JSON.parse(first.text) };
        } catch {
          return { result: first.text };
        }
      }
      return { result: raw };
    } catch (err) {
      return { error: String(err) };
    }
  }
  if (call.name === 'mcp_sum') {
    // Direct-FC wrapper around MCP get-sum. Lets the model bypass the
    // load_skill indirection by calling a flat tool with scalar params.
    const a = Number(args.a);
    const b = Number(args.b);
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      return { error: 'a and b must be numbers' };
    }
    try {
      const raw = await mcpClient.callTool('get-sum', { a, b });
      const env = raw as { content?: Array<{ type?: string; text?: string }> } | undefined;
      const text = env?.content?.[0]?.text;
      if (typeof text === 'string') {
        // Server-everything's get-sum returns "The sum of <a> and <b> is <sum>." as text.
        return { result: text };
      }
      return { result: raw };
    } catch (err) {
      return { error: String(err) };
    }
  }
  if (call.name === 'mcp_echo') {
    const message = String(args.message ?? '');
    if (!message) {
      return { error: 'message required' };
    }
    try {
      const raw = await mcpClient.callTool('echo', { message });
      const env = raw as { content?: Array<{ type?: string; text?: string }> } | undefined;
      const text = env?.content?.[0]?.text;
      if (typeof text === 'string') {
        return { result: text };
      }
      return { result: raw };
    } catch (err) {
      return { error: String(err) };
    }
  }
  if (call.name === 'run_intent') {
    // Real OS-level dispatch via React Native's Linking. We honor only the
    // intents that have a stable iOS URL scheme:
    //   send_email → mailto:
    //   open_app   → known first-party schemes (Calendar, Settings, Maps, …)
    // set_alarm has no public iOS API and returns an explicit "unsupported"
    // result so the model can apologize honestly instead of pretending.
    const intent = (args.intent as string) ?? '';
    // Match Edge Gallery: `parameters` is a JSON-encoded string (Kotlin
    // SDK schema flattens nested types to String for the same reason —
    // constrained decoding fills flat slots, not objects).
    let parameters: Record<string, unknown> = {};
    const rawParams = args.parameters;
    if (typeof rawParams === 'string' && rawParams.trim()) {
      try {
        const parsed = JSON.parse(rawParams);
        if (parsed && typeof parsed === 'object') parameters = parsed as Record<string, unknown>;
      } catch (err) {
        return { intent, status: 'failed', error: `parameters JSON parse failed: ${String(err)}`, parameters: rawParams };
      }
    } else if (rawParams && typeof rawParams === 'object') {
      parameters = rawParams as Record<string, unknown>;
    }
    if (intent === 'send_email') {
      const to = String(parameters.to ?? '');
      const subject = encodeURIComponent(String(parameters.subject ?? ''));
      const body = encodeURIComponent(String(parameters.body ?? ''));
      if (!to) return { intent, parameters, status: 'failed', error: 'recipient required' };
      const url = `mailto:${to}?subject=${subject}&body=${body}`;
      try {
        await Linking.openURL(url);
        return { intent, parameters, status: 'opened mail composer', url };
      } catch (err) {
        return { intent, parameters, status: 'failed', error: String(err), url };
      }
    }
    if (intent === 'set_alarm') {
      return {
        intent,
        parameters,
        status: 'unsupported',
        error: 'iOS does not expose a public URL scheme to programmatically set alarms; the user must open the Clock app manually.',
      };
    }
    if (intent === 'open_app') {
      const appName = String(parameters.appName ?? '').toLowerCase().trim();
      const SCHEMES: Record<string, string> = {
        calendar: 'calshow://',
        maps: 'maps://',
        settings: 'App-prefs:',
        mail: 'mailto:',
        messages: 'sms:',
        facetime: 'facetime://',
        phone: 'tel:',
        reminders: 'x-apple-reminderkit://',
      };
      const url = SCHEMES[appName];
      if (!url) {
        return {
          intent,
          parameters,
          status: 'unsupported',
          error: `iOS does not expose a public URL scheme for "${appName}". Supported app names: ${Object.keys(SCHEMES).join(', ')}.`,
        };
      }
      try {
        await Linking.openURL(url);
        return { intent, parameters, status: 'opened', url, appName };
      } catch (err) {
        return { intent, parameters, status: 'failed', error: String(err), url };
      }
    }
    return { intent, parameters, status: 'unknown intent', error: `unsupported intent: ${intent}` };
  }
  return { error: `unknown tool: ${call.name}` };
}

interface ScenarioResult {
  name: string;
  prompt: string;
  passed: boolean;
  toolCalled?: string;
  toolArgs?: Record<string, unknown>;
  rawFirstResponse?: string;
  finalText: string;
  durationMs: number;
  error?: string;
  stats?: { tps: number; completion: number; ttft: number; total: number };
}

type LogLevel = 'info' | 'pass' | 'fail';
interface LogEntry {
  level: LogLevel;
  message: string;
}

function App(): React.JSX.Element {
  const isDark = useColorScheme() === 'dark';
  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} />
      <AppContent />
    </SafeAreaProvider>
  );
}

interface SharedRuntime {
  llm: ReturnType<typeof createLLM>;
  ensureModel: (target: 'chat' | 'chat-direct') => Promise<void>;
  // Reads the most recent backend_resolved from the native stderr log.
  // Refreshed after every loadModel call (engine init writes the marker).
  getBackend: () => Promise<'GPU' | 'CPU' | 'NPU' | 'unknown'>;
}

function AppContent(): React.JSX.Element {
  const insets = useSafeAreaInsets();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [runtime, setRuntime] = useState<SharedRuntime | null>(null);
  // chatModelKey is locked in once the user clicks "Start" in ModelGate. After
  // that the harness loads with the chosen filename and runtime takes over.
  const [chatModelKey, setChatModelKey] = useState<string | null>(null);
  // Polled backend (GPU/CPU/NPU). The runHarness phase blocks the UI for many
  // seconds before runtime/ChatPanel appears, so we poll the native stderr
  // log here too so the user sees what backend is in use the whole time.
  const [backend, setBackend] = useState<'GPU' | 'CPU' | 'NPU' | 'unknown'>('unknown');
  const startedRef = useRef(false);
  const log = (level: LogLevel, message: string) =>
    setLogs((prev) => [...prev, { level, message }]);

  useEffect(() => {
    if (!chatModelKey) return;
    let cancelled = false;
    const tick = async () => {
      const b = await readResolvedBackend();
      if (!cancelled) setBackend(b);
    };
    void tick();
    const id = setInterval(tick, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, [chatModelKey]);

  useEffect(() => {
    if (!chatModelKey || startedRef.current) return;
    startedRef.current = true;
    const chosen = MODEL_CATALOG.find((m) => m.key === chatModelKey);
    if (!chosen) {
      log('fail', `unknown chat model key: ${chatModelKey}`);
      return;
    }
    void runHarness(log, setRuntime, chosen.filename);
  }, [chatModelKey]);

  // resetModel — return to ModelGate so the user can switch between E2B/E4B
  // without rebooting the app. Releases the native engine first (close()) so
  // the next loadModel starts from a clean slate; otherwise both models would
  // sit in RAM until the JS GC happens to release the native handle.
  const resetModel = () => {
    try {
      runtime?.llm.close();
    } catch {}
    setRuntime(null);
    setChatModelKey(null);
    setLogs([]);
    startedRef.current = false;
  };

  // ModelGate runs first: shows the catalog, lets the user download missing
  // models, and on "Start" hands control to runHarness with the chosen file.
  // Once the harness completes and runtime is handed back, demote the log to
  // a small collapsed strip and let the chat take the full remaining height.
  return (
    <View style={[styles.container, { paddingTop: insets.top, paddingBottom: insets.bottom }]}>
      <Text style={styles.title}>
        rn-litert-gemma4 · Step D (real FST){' '}
        {chatModelKey ? <Text style={backendStyle(backend)}>[{backend}]</Text> : null}
      </Text>
      {!chatModelKey ? (
        <ModelGate onStart={(key) => setChatModelKey(key)} />
      ) : (
        <>
          <ScrollView style={runtime ? styles.logCollapsed : styles.log}>
            {logs.map((entry, idx) => (
              <Text key={idx} style={[styles.logLine, lineStyle(entry.level)]}>
                {prefix(entry.level)} {entry.message}
              </Text>
            ))}
          </ScrollView>
          {runtime ? (
            <ChatPanel
              runtime={runtime}
              log={log}
              currentModelKey={chatModelKey}
              onChangeModel={resetModel}
            />
          ) : null}
        </>
      )}
    </View>
  );
}

// =============================================================================
// ModelGate — pre-harness model picker + downloader
// =============================================================================
function ModelGate({ onStart }: { onStart: (chatKey: string) => void }): React.JSX.Element {
  const [chatKey, setChatKey] = useState<string>(DEFAULT_CHAT_KEY);
  // existsMap[filename] = absolute path on disk OR '' if missing.
  // Re-checked after each download. Refreshed on mount.
  const [existsMap, setExistsMap] = useState<Record<string, string>>({});
  const [download, setDownload] = useState<Record<string, DownloadState>>({});
  const refreshExists = async () => {
    const out: Record<string, string> = {};
    for (const m of MODEL_CATALOG) {
      out[m.filename] = await findModelPath(m.filename);
    }
    setExistsMap(out);
  };
  useEffect(() => {
    void refreshExists();
  }, []);

  const startDownload = async (entry: ModelEntry) => {
    if (download[entry.filename]?.inProgress) return;
    const dest = `${RNFS.DocumentDirectoryPath}/${entry.filename}`;
    setDownload((prev) => ({
      ...prev,
      [entry.filename]: { ...initialDownload, inProgress: true, totalBytes: entry.sizeBytes },
    }));
    try {
      const job = RNFS.downloadFile({
        fromUrl: entry.url,
        toFile: dest,
        // Throttle progress callbacks to ~1% increments so React re-renders
        // stay reasonable on 3 GB downloads.
        progressDivider: 1,
        progress: (res) => {
          setDownload((prev) => ({
            ...prev,
            [entry.filename]: {
              inProgress: true,
              bytesWritten: res.bytesWritten,
              totalBytes: res.contentLength || entry.sizeBytes,
              jobId: job.jobId,
            },
          }));
        },
      });
      const result = await job.promise;
      if (result.statusCode !== 200) {
        throw new Error(`HTTP ${result.statusCode}`);
      }
      setDownload((prev) => ({
        ...prev,
        [entry.filename]: { ...initialDownload, bytesWritten: entry.sizeBytes, totalBytes: entry.sizeBytes },
      }));
      await refreshExists();
    } catch (err) {
      setDownload((prev) => ({
        ...prev,
        [entry.filename]: { ...initialDownload, error: String(err) },
      }));
      // Best-effort cleanup of any partial file so retry starts clean.
      try { await RNFS.unlink(dest); } catch {}
    }
  };

  const cancelDownload = (filename: string) => {
    const j = download[filename]?.jobId;
    if (j !== undefined) {
      RNFS.stopDownload(j);
    }
    setDownload((prev) => ({ ...prev, [filename]: initialDownload }));
  };

  const chatEntry = MODEL_CATALOG.find((m) => m.key === chatKey);
  const chatReady = chatEntry ? !!existsMap[chatEntry.filename] : false;
  const canStart = chatReady;
  const renderRow = (entry: ModelEntry) => {
    const exists = !!existsMap[entry.filename];
    const dl = download[entry.filename];
    const pct = dl && dl.totalBytes > 0 ? Math.floor((dl.bytesWritten / dl.totalBytes) * 100) : 0;
    return (
      <View key={entry.filename} style={styles.gateRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.gateRowLabel}>{entry.label}</Text>
          <Text style={styles.gateRowSub}>
            {entry.filename} · {formatMB(entry.sizeBytes)}
          </Text>
          {dl?.inProgress ? (
            <Text style={styles.gateRowSub}>
              downloading… {pct}% ({formatMB(dl.bytesWritten)} / {formatMB(dl.totalBytes)})
            </Text>
          ) : null}
          {dl?.error ? <Text style={styles.gateRowError}>error: {dl.error}</Text> : null}
        </View>
        {exists ? (
          <View style={[styles.gatePill, styles.gatePillReady]}>
            <Text style={styles.gatePillText}>ready</Text>
          </View>
        ) : dl?.inProgress ? (
          <TouchableOpacity onPress={() => cancelDownload(entry.filename)} style={[styles.gatePill, styles.gatePillCancel]}>
            <Text style={styles.gatePillText}>cancel</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity onPress={() => startDownload(entry)} style={[styles.gatePill, styles.gatePillDownload]}>
            <Text style={styles.gatePillText}>download</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  };

  return (
    <ScrollView style={styles.gateContainer}>
      <Text style={styles.gateHeader}>Pick chat model</Text>
      <View style={styles.gateChoices}>
        {MODEL_CATALOG.map((m) => (
          <TouchableOpacity
            key={m.key}
            onPress={() => setChatKey(m.key)}
            style={[styles.gateChoice, chatKey === m.key && styles.gateChoiceActive]}
          >
            <Text style={[styles.gateChoiceText, chatKey === m.key && styles.gateChoiceTextActive]}>
              {m.key}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
      <Text style={styles.gateHeader}>Files</Text>
      {chatEntry ? renderRow(chatEntry) : null}
      <TouchableOpacity
        onPress={() => onStart(chatKey)}
        style={[styles.gateStartBtn, !canStart && styles.gateStartBtnDisabled]}
        disabled={!canStart}
      >
        <Text style={styles.gateStartBtnText}>
          {canStart ? `Start with ${chatEntry?.key ?? '?'}` : 'Download chat model first'}
        </Text>
      </TouchableOpacity>
      <Text style={styles.gateHint}>
        Tip: GPU is requested for chat models on iOS sim. The harness emits
        `[FORK] backend_resolved=…` to the litert-stderr.log so you can confirm
        whether GPU was actually used or it fell back to CPU.
      </Text>
    </ScrollView>
  );
}

interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
  trace?: string;
}

function ChatPanel({
  runtime,
  log,
  currentModelKey,
  onChangeModel,
}: {
  runtime: SharedRuntime;
  log: (level: LogLevel, msg: string) => void;
  currentModelKey: string;
  onChangeModel: () => void;
}): React.JSX.Element {
  const [mode, setMode] = useState<'skill' | 'direct'>('skill');
  // Resolved backend (read from native stderr log). Polled on mount + after
  // every send so a model switch reflects the latest result.
  const [backend, setBackend] = useState<'GPU' | 'CPU' | 'NPU' | 'unknown'>('unknown');
  useEffect(() => {
    void runtime.getBackend().then(setBackend);
  }, [runtime]);
  const [input, setInput] = useState('');
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const lastTargetRef = useRef<'chat' | 'chat-direct' | null>(null);

  const send = async () => {
    const userMsg = input.trim();
    if (!userMsg || busy) return;
    setBusy(true);
    setInput('');
    setHistory((prev) => [...prev, { role: 'user', text: userMsg }]);
    try {
      const target: 'chat' | 'chat-direct' = mode === 'direct' ? 'chat-direct' : 'chat';
      await runtime.ensureModel(target);
      // Only reset when the mode actually changes — otherwise prior turns stay
      // in conversation context so the model can do follow-ups like "what
      // about this week?" after a city was already mentioned.
      if (lastTargetRef.current !== null && lastTargetRef.current !== target) {
        runtime.llm.resetConversation();
      }
      lastTargetRef.current = target;
      let raw = await runtime.llm.sendMessage(userMsg);
      const trace: string[] = [];
      const MAX_HOPS = 6;
      for (let hop = 0; hop < MAX_HOPS; hop++) {
        const call = parseGemma4ToolCall(raw);
        if (!call) break;
        trace.push(call.name);
        const result = await executeTool(call);
        const toolResp = buildGemma4ToolResponse(call.name, result);
        raw = await runtime.llm.sendMessage(toolResp);
      }
      // Cleanup: strip native control-token leftovers + detect truncated tool
      // calls (model hit max_output_tokens mid-call → no `<tool_call|>` end
      // marker → parser returned null, leaving raw markers in output).
      let text = raw;
      const truncatedMarker = text.includes('<|tool_call>') && !text.includes('<tool_call|>');
      if (truncatedMarker) {
        text = '[model output truncated mid-tool-call — try a more specific prompt or a different mode]';
      } else if (parseGemma4ToolCall(raw)) {
        text = '[tool-loop hit MAX_HOPS]';
      } else {
        // Strip Gemma 4 chat-template control tokens that occasionally leak
        // (`<|"|>`, `<end_of_turn>`, …) so the user sees clean text.
        text = text
          .replace(/<\|"\|>/g, '')
          .replace(/<\|tool_call>/g, '')
          .replace(/<tool_call\|>/g, '')
          .replace(/<end_of_turn>/g, '')
          .replace(/<start_of_turn>(?:model|user)?/g, '')
          .trim();
      }
      setHistory((prev) => [
        ...prev,
        { role: 'assistant', text, trace: trace.length ? trace.join(' → ') : undefined },
      ]);
    } catch (err) {
      log('fail', `chat error: ${String(err)}`);
      setHistory((prev) => [...prev, { role: 'assistant', text: `[error: ${String(err)}]` }]);
    } finally {
      setBusy(false);
      // Re-poll resolved backend in case ensureModel triggered a fresh
      // loadModel (mode switch chat ↔ chat-direct, or first call).
      void runtime.getBackend().then(setBackend);
    }
  };

  return (
    <View style={styles.chatPanel}>
      <View style={styles.chatHeaderRow}>
        <Text style={styles.chatModelLabel}>
          model: {currentModelKey}{' '}
          <Text style={backendStyle(backend)}>[{backend}]</Text>
        </Text>
        <TouchableOpacity onPress={onChangeModel} style={styles.chatChangeBtn}>
          <Text style={styles.chatChangeBtnText}>↻ change model</Text>
        </TouchableOpacity>
      </View>
      <View style={styles.chatModeRow}>
        <TouchableOpacity
          accessibilityRole="button"
          onPress={() => setMode('skill')}
          style={[styles.modeBtn, mode === 'skill' && styles.modeBtnActive]}
        >
          <Text style={[styles.modeBtnText, mode === 'skill' && styles.modeBtnTextActive]}>load_skill</Text>
        </TouchableOpacity>
        <TouchableOpacity
          accessibilityRole="button"
          onPress={() => setMode('direct')}
          style={[styles.modeBtn, mode === 'direct' && styles.modeBtnActive]}
        >
          <Text style={[styles.modeBtnText, mode === 'direct' && styles.modeBtnTextActive]}>direct FC</Text>
        </TouchableOpacity>
      </View>
      <ScrollView style={styles.chatHistory}>
        {history.map((t, i) => (
          <View key={i} style={[styles.chatBubble, t.role === 'user' ? styles.chatUser : styles.chatAssistant]}>
            {t.trace ? <Text style={styles.chatTrace}>tools: {t.trace}</Text> : null}
            <Text style={styles.chatText}>{t.text}</Text>
          </View>
        ))}
        {busy ? <Text style={styles.chatBusy}>thinking…</Text> : null}
      </ScrollView>
      <View style={styles.chatInputRow}>
        <TextInput
          style={styles.chatInput}
          value={input}
          onChangeText={setInput}
          placeholder={`type a message (${mode})`}
          placeholderTextColor="#666"
          editable={!busy}
          onSubmitEditing={send}
        />
        <TouchableOpacity
          accessibilityRole="button"
          style={[styles.sendBtn, busy && styles.sendBtnDisabled]}
          disabled={busy}
          onPress={send}
        >
          <Text style={styles.sendBtnText}>{busy ? '…' : 'Send'}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

async function runHarness(
  log: (level: LogLevel, msg: string) => void,
  setRuntime: (r: SharedRuntime) => void,
  chatModelFilename: string,
): Promise<void> {
  const CHAT_MODEL_FILENAME = chatModelFilename;
  const startedAt = Date.now();
  const scenarios: ScenarioResult[] = [];
  const writeResults = async (status: 'running' | 'done' | 'failed', error?: string) => {
    const out = {
      status,
      startedAt,
      finishedAt: Date.now(),
      durationMs: Date.now() - startedAt,
      scenarios,
      error,
    };
    await RNFS.writeFile(
      `${RNFS.DocumentDirectoryPath}/${RESULTS_FILE}`,
      JSON.stringify(out, null, 2),
      'utf8',
    );
  };

  try {
    await writeResults('running');
    log('info', `─── Single-model setup: ${CHAT_MODEL_FILENAME} (Edge Gallery agent_chat pattern) ───`);
    const chatPath = await findModelPath(CHAT_MODEL_FILENAME);
    log('info', `chat model: ${chatPath || 'MISSING'}`);
    if (!chatPath) {
      throw new Error(`Chat model not found. Place ${CHAT_MODEL_FILENAME} under Documents/ or Documents/models/.`);
    }
    const llm = createLLM();
    let currentModel: 'chat' | 'chat-direct' | null = null;
    const tLoad = Date.now();
    // Edge Gallery load_skill pattern adapted with general-chat escape clause.
    // Skills list is dynamically built from SKILLS registry — to add a new
    // skill, just push another entry; the model sees it on the next load.
    const skillsList = SKILLS.map((s) => `   - ${s.name}: ${s.description}`).join('\n');
    const systemPrompt = [
      'You are an AI assistant that helps users using skills. For EVERY new user message, you MUST execute the following steps in exact order. You MUST NOT skip any steps.',
      '',
      'CRITICAL RULE: You MUST execute all steps silently. Do NOT generate or output any internal thoughts, reasoning, explanations, or intermediate text at ANY step.',
      '',
      '1. First, find the most relevant skill from the following list:',
      '',
      skillsList,
      '',
      '   If you are unsure which skill fits, choose `general_chat`.',
      '',
      '2. Call the `load_skill` tool with skillName=<chosen> immediately. You MUST NOT emit any other tool or any text reply at this step. The ONLY valid output is a load_skill tool call.',
      '',
      '3. Once load_skill returns instructions, follow those instructions exactly. The instructions will tell you whether to call run_api, run_mcp_tool, or simply respond directly to the user. Do NOT improvise different tool calls.',
      '',
      '4. If a run_api / run_mcp_tool call returns data, form the final user-facing reply per the skill instructions.',
      '',
      'FORBIDDEN: Apologizing about lack of real-time data instead of calling load_skill. Refusing to call load_skill. Replying with text directly without first calling load_skill. Promising to call a tool later.',
    ].join('\n');

    // Pure positive few-shot: show TWO consecutive successful tool-call
    // exchanges. No refusal/correction — those backfired (model copies the
    // wrong attempt). Only correct behavior in history.
    const priorMessages = JSON.stringify([
      { role: 'user', content: 'What is the temperature in Madrid?' },
      {
        role: 'assistant',
        tool_calls: [
          { function: { name: 'get_weather', arguments: { city: 'Madrid' } } },
        ],
      },
      {
        role: 'tool',
        content: [
          { name: 'get_weather', response: { city: 'Madrid', temperatureC: 17 } },
        ],
      },
      { role: 'assistant', content: 'It is 17°C in Madrid right now.' },
      { role: 'user', content: 'How about Paris weather?' },
      {
        role: 'assistant',
        tool_calls: [
          { function: { name: 'get_weather', arguments: { city: 'Paris' } } },
        ],
      },
      {
        role: 'tool',
        content: [
          { name: 'get_weather', response: { city: 'Paris', temperatureC: 14 } },
        ],
      },
      { role: 'assistant', content: 'It is 14°C in Paris right now.' },
      { role: 'user', content: 'Calculate 100 plus 200 exactly.' },
      {
        role: 'assistant',
        tool_calls: [
          { function: { name: 'add_numbers', arguments: { a: 100, b: 200 } } },
        ],
      },
      {
        role: 'tool',
        content: [{ name: 'add_numbers', response: { result: 300 } }],
      },
      { role: 'assistant', content: 'The result is 300.' },
    ]);

    void priorMessages;
    void tLoad;
    // ensureModel(target): hot-swaps the loaded model when the requested
    // target differs from the currently loaded one. Closes + loadModel(),
    // which the wrapper handles via internal close() before re-init.
    // Minimal direct-FC system prompt: no load_skill mandate, no skills list.
    // Just expose the tools and let the model pick. This baseline mirrors how
    // a developer would expose MCP/native tools without indirection.
    const directSystemPrompt = [
      'You are an AI assistant with access to a small set of tools.',
      '',
      'When a user request maps to one of the tools, call that tool directly with the appropriate arguments. When no tool fits, answer concisely from your own knowledge.',
      '',
      'CRITICAL: do not refuse action requests by saying "I cannot" or "I do not have access". If a tool exists for the request, call it. If not, give the best brief answer you can or admit uncertainty in one sentence.',
    ].join('\n');

    const ensureModel = async (target: 'chat' | 'chat-direct'): Promise<void> => {
      if (currentModel === target) return;
      log('info', `─── Loading ${target} mode (${CHAT_MODEL_FILENAME}) ───`);
      const t0 = Date.now();
      // Two modes share ONE Gemma 4 weight file:
      //   chat        — load_skill MUST-execute pattern + TOOLS
      //   chat-direct — minimal direct-FC system prompt + DIRECT_TOOLS
      // Both use GPU + constrained decoding (Edge Gallery upstream pattern).
      // CD enforces structured tool-call tokens (FST). We tested CD=false to
      // measure speculative decoding impact — drafted/verified stayed at 0
      // even without CD, confirming spec_dec is unusable on CPU regardless.
      // Spec dec only kicks in on real device GPU.
      if (target === 'chat-direct') {
        await llm.loadModel(chatPath, {
          backend: 'gpu',
          tools: JSON.stringify(DIRECT_TOOLS),
          enableConstrainedDecoding: true,
          systemPrompt: directSystemPrompt,
        });
      } else {
        await llm.loadModel(chatPath, {
          backend: 'gpu',
          tools: JSON.stringify(TOOLS),
          enableConstrainedDecoding: true,
          systemPrompt,
        });
      }
      currentModel = target;
      log('pass', `${target} mode loaded in ${Date.now() - t0} ms`);
    };

    // expectMatch: optional substring(s) the final reply must contain to pass.
    // Tightens validation beyond the load_skill name match — catches cases
    // where the right skill is chosen but the actual tool round-trip fails
    // (e.g. MCP returns an error and the model parrots "Error" back).
    // mode 'skill'  : load_skill mandatory pattern (TOOLS + load_skill prompt)
    // mode 'direct' : direct FC pattern (DIRECT_TOOLS + minimal prompt). The
    //                 model calls e.g. mcp_sum directly; expectSkill/skillCalled
    //                 is interpreted as the first tool name in this mode.
    const scenarioDefs: { name: string; prompt: string; expectSkill: string | null; rule: string; expectMatch?: string[]; expectMissing?: string[]; mode?: 'skill' | 'direct' }[] = [
      // FC mobile-action scenarios (real OS dispatch via Linking, not stubs)
      // M1.email on simulator: mailto: launch may fail because the sim has
      // no Mail account configured. On a real device the same call opens the
      // Mail composer. We accept either the success message ("composer") or
      // the explicit failure surfaced from Linking ("Unable to open URL").
      { name: 'M1.email', rule: 'load_skill → mailto: open (sim may fail)', prompt: 'Send an email to bob@example.com saying "Meeting at 3pm".', expectSkill: 'send_email', expectMatch: ['bob@example.com'] },
      { name: 'M2.alarm', rule: 'load_skill → unsupported', prompt: 'Set an alarm for 7:00 AM tomorrow labeled "wake up".', expectSkill: 'set_alarm', expectMatch: ['Clock', 'cannot'] },
      { name: 'M3.openapp', rule: 'load_skill → calshow://', prompt: 'Open the Calendar app.', expectSkill: 'open_app', expectMatch: ['Calendar'], expectMissing: ['unsupported', 'cannot'] },
      { name: 'M4.weather', rule: 'load_skill → weather', prompt: "What's the temperature in Seoul right now?", expectSkill: 'weather', expectMatch: ['Seoul'] },
      // Chat scenarios — under load_skill MUST-execute pattern, every prompt
      // dispatches through load_skill first.
      //   B1 -> general_chat: greeting (no external lookup)
      //   B2 -> wikipedia: niche topic prefixed with "On Wikipedia," to force
      //     the wikipedia skill (verifies run_api Wikipedia REST flow end-to-end)
      //   C1 -> general_chat: subjective/unknowable, model admits it
      { name: 'B1.greet', rule: 'load_skill → general_chat', prompt: 'Hi, how are you today?', expectSkill: 'general_chat' },
      { name: 'B2.wiki', rule: 'load_skill → wikipedia → run_api', prompt: 'On Wikipedia, look up the Korean modernist poet Yi Sang and summarize his life.', expectSkill: 'wikipedia', expectMatch: ['Yi Sang'] },
      { name: 'C1.unknown', rule: 'load_skill → general_chat (admit unknown)', prompt: 'What did I have for lunch yesterday?', expectSkill: 'general_chat', expectMissing: ['Error', 'mcp error'] },
      // MCP scenarios — exercise real streamable-HTTP MCP protocol against
      // @modelcontextprotocol/server-everything (running on host:4567).
      // expectMatch enforces actual round-trip success (sum=100, echoed text)
      // — without this the model could parrot "Error" or stub-success and
      // still register PASS based only on the load_skill name match.
      { name: 'D1.mcp_sum', rule: 'load_skill → run_mcp_tool(get-sum)', prompt: 'Use MCP to add 47 and 53 exactly.', expectSkill: 'mcp_sum', expectMatch: ['100'], expectMissing: ['Error', 'error'] },
      { name: 'D2.mcp_echo', rule: 'load_skill → run_mcp_tool(echo)', prompt: 'Use MCP echo to repeat the message: hello world', expectSkill: 'mcp_echo', expectMatch: ['hello world'], expectMissing: ['Error', 'error'] },
      // ── Direct-FC scenarios (no load_skill mediation; same Gemma 4 model,
      //    DIRECT_TOOLS list with flat scalar params, minimal system prompt).
      //    Validates that direct tool calling also works alongside the
      //    load_skill pattern. expectSkill here = expected first tool name.
      { name: 'E1.direct_weather', rule: 'direct → get_weather', prompt: "What's the temperature in Seoul right now?", expectSkill: 'get_weather', expectMatch: ['Seoul'], mode: 'direct' },
      { name: 'E2.direct_mcp_sum', rule: 'direct → mcp_sum', prompt: 'Use MCP to add 47 and 53 exactly.', expectSkill: 'mcp_sum', expectMatch: ['100'], expectMissing: ['Error', 'error'], mode: 'direct' },
      { name: 'E3.direct_mcp_echo', rule: 'direct → mcp_echo', prompt: 'Use MCP echo to repeat the message: hello world', expectSkill: 'mcp_echo', expectMatch: ['hello world'], expectMissing: ['Error', 'error'], mode: 'direct' },
      { name: 'E4.direct_chat', rule: 'direct → no tool', prompt: 'Hi, how are you today?', expectSkill: null, mode: 'direct' },
    ];

    for (const def of scenarioDefs) {
      const sStart = Date.now();
      const route: 'chat' | 'chat-direct' =
        def.mode === 'direct' ? 'chat-direct' : 'chat';
      log('info', `▶ ${def.name} [route=${route} mode=${def.mode ?? 'skill'}]: ${def.prompt}`);
      try {
        await ensureModel(route);
        // Reset conversation between scenarios so prior tool-call context
        // does not bias the next prompt — without this, the model trained
        // by recent successful tool calls starts emitting raw skill names
        // as text answers instead of going through load_skill.
        llm.resetConversation();
        // Multi-turn agent loop: load_skill → run_api → ... → final reply
        // Cap at 6 hops to prevent infinite loops.
        const MAX_HOPS = 6;
        const trace: { tool: string; args: unknown }[] = [];
        let firstCall: Gemma4ToolCall | null = null;
        let raw = await llm.sendMessage(def.prompt);
        let finalText = raw;
        const firstStats = llm.getStats();
        for (let hop = 0; hop < MAX_HOPS; hop++) {
          // Both chat and chat-direct use the same Gemma 4 weights, so a single
          // parser handles every output (structured tool_calls JSON, native
          // <|tool_call> tokens, or JSON fence — see parseGemma4ToolCall).
          const call: Gemma4ToolCall | null = parseGemma4ToolCall(raw);
          if (!call) {
            finalText = raw;
            break;
          }
          if (hop === 0) firstCall = call;
          trace.push({ tool: call.name, args: call.arguments });
          log('info', `  hop ${hop}: ${call.name}(${JSON.stringify(call.arguments).slice(0, 100)})`);
          const result = await executeTool(call);
          log('info', `    → ${JSON.stringify(result).slice(0, 120)}`);
          const toolResp = buildGemma4ToolResponse(call.name, result);
          raw = await llm.sendMessage(toolResp);
          finalText = raw;
        }
        const stats = firstStats;
        // For 'skill' mode the first call should be `load_skill(<name>)` and
        // we compare the chosen skill name. For 'direct' mode the model is
        // expected to call the target tool directly (e.g. `mcp_sum`), so we
        // compare against the first tool name itself.
        const isDirect = def.mode === 'direct';
        const skillCalled = isDirect
          ? (firstCall?.name ?? null)
          : firstCall?.name === 'load_skill'
            ? ((firstCall.arguments?.skillName as string) ?? '')
            : null;
        const skillOk =
          def.expectSkill === null
            ? firstCall === null
            : skillCalled === def.expectSkill;
        const matchOk = (def.expectMatch ?? []).every((needle) =>
          finalText.toLowerCase().includes(needle.toLowerCase()),
        );
        const missingOk = (def.expectMissing ?? []).every(
          (needle) => !finalText.toLowerCase().includes(needle.toLowerCase()),
        );
        const passed = skillOk && matchOk && missingOk;
        const traceSummary = trace.map((t) => t.tool).join(' → ') || 'no-tool';
        scenarios.push({
          name: def.name,
          prompt: def.prompt,
          passed,
          ...(firstCall ? { toolCalled: firstCall.name, toolArgs: firstCall.arguments } : {}),
          rawFirstResponse: traceSummary + ' | ' + finalText.slice(0, 300),
          finalText: finalText.slice(0, 400),
          durationMs: Date.now() - sStart,
          stats: {
            tps: stats.tokensPerSecond,
            completion: stats.completionTokens,
            ttft: stats.timeToFirstToken,
            total: stats.totalTime,
          },
        });
        log(passed ? 'pass' : 'fail',
          `${def.name} ${passed ? 'PASS' : 'FAIL'} | trace=${traceSummary} | skill=${skillCalled ?? 'none'} | ${Date.now() - sStart}ms`);
        log('info', `  final: ${finalText.slice(0, 200)}${finalText.length > 200 ? '…' : ''}`);
      } catch (err) {
        scenarios.push({
          name: def.name,
          prompt: def.prompt,
          passed: false,
          finalText: '',
          durationMs: Date.now() - sStart,
          error: String(err),
        });
        log('fail', `${def.name} ERROR: ${String(err)}`);
      }
      await writeResults('running');
    }

    const passed = scenarios.filter((s) => s.passed).length;
    log(passed === scenarios.length ? 'pass' : 'info',
      `═══ ${passed}/${scenarios.length} scenarios passed ═══`);
    await writeResults('done');
    // Hand the live LLM + ensureModel callback to AppContent so the
    // interactive ChatPanel can reuse the same loaded model (no second
    // 2.4GB load).
    setRuntime({ llm, ensureModel, getBackend: readResolvedBackend });
  } catch (err) {
    log('fail', `harness error: ${String(err)}`);
    await writeResults('failed', String(err));
  }
}

function prefix(level: LogLevel): string {
  if (level === 'pass') return '✓';
  if (level === 'fail') return '✗';
  return '·';
}

function lineStyle(level: LogLevel) {
  if (level === 'pass') return { color: '#1a7f37' };
  if (level === 'fail') return { color: '#cf222e' };
  return { color: '#1f2328' };
}

function backendStyle(b: 'GPU' | 'CPU' | 'NPU' | 'unknown') {
  if (b === 'GPU') return { color: '#1a7f37', fontWeight: '700' as const };
  if (b === 'CPU') return { color: '#bf8700', fontWeight: '700' as const };
  if (b === 'NPU') return { color: '#0969da', fontWeight: '700' as const };
  return { color: '#6e7781', fontWeight: '700' as const };
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff', padding: 12 },
  title: { fontSize: 16, fontWeight: '600', marginBottom: 8 },
  log: { flex: 1, backgroundColor: '#f6f8fa', padding: 6, borderRadius: 6 },
  logCollapsed: {
    maxHeight: 80,
    backgroundColor: '#f6f8fa',
    padding: 6,
    borderRadius: 6,
    marginBottom: 8,
  },
  logLine: { fontFamily: 'Menlo', fontSize: 10, marginBottom: 2 },
  chatPanel: { flex: 1, marginTop: 4, borderTopWidth: 1, borderTopColor: '#e1e4e8', paddingTop: 12 },
  chatHeaderRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 },
  chatModelLabel: { fontSize: 11, color: '#586069', fontFamily: 'Menlo' },
  chatChangeBtn: { paddingVertical: 4, paddingHorizontal: 8, borderRadius: 4, backgroundColor: '#f0f3f6', borderWidth: 1, borderColor: '#d1d5da' },
  chatChangeBtnText: { fontSize: 11, color: '#0366d6', fontWeight: '600' },
  chatModeRow: { flexDirection: 'row', gap: 8, marginBottom: 8 },
  modeBtn: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 6,
    backgroundColor: '#f0f3f6',
    alignItems: 'center',
  },
  modeBtnActive: { backgroundColor: '#0366d6' },
  modeBtnText: { fontSize: 12, fontWeight: '600', color: '#586069' },
  modeBtnTextActive: { color: '#fff' },
  chatHistory: { flex: 1, marginBottom: 8 },
  chatBubble: { padding: 8, borderRadius: 8, marginBottom: 6 },
  chatUser: { backgroundColor: '#dbedff', alignSelf: 'flex-end', maxWidth: '80%' },
  chatAssistant: { backgroundColor: '#f6f8fa', alignSelf: 'flex-start', maxWidth: '90%' },
  chatTrace: { fontFamily: 'Menlo', fontSize: 9, color: '#6a737d', marginBottom: 2 },
  chatText: { fontSize: 13, color: '#24292e' },
  chatBusy: { fontSize: 11, color: '#6a737d', fontStyle: 'italic', textAlign: 'center', marginVertical: 4 },
  chatInputRow: { flexDirection: 'row', gap: 6, alignItems: 'center' },
  chatInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#d1d5da',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 13,
    color: '#24292e',
  },
  sendBtn: { backgroundColor: '#0366d6', paddingVertical: 8, paddingHorizontal: 14, borderRadius: 6 },
  sendBtnDisabled: { backgroundColor: '#959da5' },
  sendBtnText: { color: '#fff', fontWeight: '600', fontSize: 13 },
  // ModelGate
  gateContainer: { flex: 1 },
  gateHeader: { fontSize: 13, fontWeight: '700', color: '#24292e', marginTop: 12, marginBottom: 6 },
  gateChoices: { flexDirection: 'row', gap: 8, flexWrap: 'wrap' },
  gateChoice: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    backgroundColor: '#f0f3f6',
    borderWidth: 1,
    borderColor: '#d1d5da',
  },
  gateChoiceActive: { backgroundColor: '#0366d6', borderColor: '#0366d6' },
  gateChoiceText: { fontSize: 12, color: '#586069', fontWeight: '600' },
  gateChoiceTextActive: { color: '#fff' },
  gateRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 10,
    borderRadius: 6,
    backgroundColor: '#f6f8fa',
    marginBottom: 6,
    gap: 8,
  },
  gateRowLabel: { fontSize: 12, fontWeight: '600', color: '#24292e' },
  gateRowSub: { fontSize: 10, color: '#586069', marginTop: 2 },
  gateRowError: { fontSize: 10, color: '#cb2431', marginTop: 2 },
  gatePill: { paddingVertical: 6, paddingHorizontal: 10, borderRadius: 12 },
  gatePillReady: { backgroundColor: '#28a745' },
  gatePillDownload: { backgroundColor: '#0366d6' },
  gatePillCancel: { backgroundColor: '#cb2431' },
  gatePillText: { color: '#fff', fontSize: 11, fontWeight: '600' },
  gateStartBtn: {
    backgroundColor: '#0366d6',
    paddingVertical: 12,
    borderRadius: 6,
    marginTop: 14,
    alignItems: 'center',
  },
  gateStartBtnDisabled: { backgroundColor: '#959da5' },
  gateStartBtnText: { color: '#fff', fontWeight: '700', fontSize: 13 },
  gateHint: { fontSize: 10, color: '#586069', marginTop: 14, lineHeight: 14 },
});

export default App;
