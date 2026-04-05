import React, { useState, useCallback, useMemo, useRef } from "react";
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Platform,
  ActivityIndicator,
  TextInput,
  Image,
  Switch,
} from "react-native";
import { SafeAreaProvider, SafeAreaView } from "react-native-safe-area-context";
import {
  useModel,
  GEMMA_3N_E2B_IT_INT4,
  GEMMA_4_E2B_IT,
  checkMultimodalSupport,
  checkBackendSupport,
  applyGemmaTemplate,
  createMemoryTracker,
  type ChatMessage,
} from "react-native-litert-lm";

// Test asset paths — resolved at runtime from bundled assets
// On iOS, Metro bundles images into the app; resolveAssetSource gives us a URI.
// On Android, the file must be pre-pushed to the device.
const TEST_IMAGE_ASSET = require("./test.jpeg");

async function getTestImagePath(modelInstance?: any): Promise<string> {
  if (Platform.OS === "android") {
    return "/data/local/tmp/test.jpeg";
  }
  // iOS: resolveAssetSource returns a file:// URI for local assets in prod,
  // but an HTTP URL in dev mode (Metro-served)
  const source = Image.resolveAssetSource(TEST_IMAGE_ASSET);
  if (source.uri.startsWith("file://")) {
    return source.uri.replace("file://", "");
  }
  // Dev mode: Metro serves via HTTP — use the model's download helper
  // to fetch it to a local cache file
  if (modelInstance?.downloadModel) {
    return await modelInstance.downloadModel(source.uri, "test_image.jpeg");
  }
  throw new Error(
    "Cannot resolve test image path in dev mode without model instance",
  );
}

const TEST_AUDIO_PATH = Platform.select({
  ios: "/tmp/test.wav", // Audio test not yet supported on iOS
  android: "/data/local/tmp/test.wav",
  default: "/tmp/test.wav",
})!;

const THEME = {
  bg: "#050505",
  card: "#121212",
  accent: "#3B82F6",
  success: "#10B981",
  warning: "#F59E0B",
  error: "#EF4444",
  text: "#F9FAF7",
  textDim: "#9CA3AF",
  border: "#262626",
};

type LogEntry = {
  timestamp: number;
  message: string;
  type: "info" | "success" | "error";
};

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(i > 1 ? 1 : 0)} ${units[i]}`;
}

export default function App() {
  return (
    <SafeAreaProvider>
      <Main />
    </SafeAreaProvider>
  );
}

function Main() {
  const [useGpu, setUseGpu] = useState(false);
  const [selectedModel, setSelectedModel] = useState<"gemma3n" | "gemma4">(
    "gemma3n",
  );

  const MODEL_OPTIONS = {
    gemma3n: { label: "Gemma 3n E2B (1.3 GB)", url: GEMMA_3N_E2B_IT_INT4 },
    gemma4: { label: "Gemma 4 E2B (2.6 GB)", url: GEMMA_4_E2B_IT },
  };

  const config = useMemo(
    () => ({
      backend: useGpu ? ("gpu" as const) : ("cpu" as const),
      systemPrompt: "You are a helpful assistant.",
      maxTokens: 1024,
      autoLoad: false,
      enableMemoryTracking: true,
      maxMemorySnapshots: 100,
    }),
    [useGpu],
  );

  const {
    model,
    isReady,
    downloadProgress,
    error,
    deleteModel,
    load,
    memoryTracker,
    memorySummary,
  } = useModel(MODEL_OPTIONS[selectedModel].url, config);

  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [lastLatency, setLastLatency] = useState<number | null>(null);
  const [tokensPerSec, setTokensPerSec] = useState<number | null>(null);
  const [chatInput, setChatInput] = useState("");

  const log = useCallback(
    (message: string, type: LogEntry["type"] = "info") => {
      console.log(`[App] ${message}`);
      setLogs((prev) => [
        { timestamp: Date.now(), message, type },
        ...prev.slice(0, 99),
      ]);
    },
    [],
  );

  const runFullTest = async () => {
    if (isRunning || !isReady) return;
    setIsRunning(true);
    setLogs([]);

    try {
      log("Starting Full Test Suite...", "info");

      // Test 1: Hook state
      log(
        `Model instance: ${!!model ? "✓" : "✗"}`,
        model ? "success" : "error",
      );
      log(`isReady: ${isReady}`, "success");

      // Test 2: Backend support
      const cpuWarn = checkBackendSupport("cpu");
      const gpuWarn = checkBackendSupport("gpu");
      const npuWarn = checkBackendSupport("npu");
      log(`CPU: ${cpuWarn ?? "OK"}`);
      log(`GPU: ${gpuWarn ?? "OK"}`);
      log(`NPU: ${npuWarn ?? "OK"}`);

      // Test 3: Multimodal
      const mmError = checkMultimodalSupport();
      log(
        mmError ? `Multimodal: ${mmError}` : "Multimodal supported",
        mmError ? "info" : "success",
      );

      // Test 4: Templates
      const hist: ChatMessage[] = [
        { role: "user", content: "Hello!" },
        { role: "model", content: "Hi!" },
        { role: "user", content: "How are you?" },
      ];
      const tpl = applyGemmaTemplate(hist, "You are helpful.");
      log(`Gemma template: ${tpl.length} chars`, "success");

      // Test 5: Inference
      if (!model) throw new Error("Model not available");
      log("Running inference...");
      const t0 = Date.now();
      const response = await model.sendMessage(
        "What is 2+2? Answer with just the number.",
      );
      const elapsed = Date.now() - t0;
      setLastLatency(elapsed);
      log(`Response: "${response}"`, "success");
      log(`Latency: ${elapsed}ms`, "success");

      // Test 6: Stats
      const stats = model.getStats();
      setTokensPerSec(stats.tokensPerSecond);
      log(
        `Speed: ${stats.tokensPerSecond.toFixed(1)} tok/s | Prompt: ${stats.promptTokens} | Completion: ${stats.completionTokens}`,
        "success",
      );

      // Test 7: Context
      await model.sendMessage("My name is TestUser.");
      const nameResp = await model.sendMessage("What is my name?");
      log(`Context test: "${nameResp}"`, "success");

      // Test 8: Reset
      model.resetConversation();
      log("Conversation reset", "success");

      // Test 9: Streaming
      let streamCount = 0;
      await new Promise<void>((resolve) => {
        model.sendMessageAsync(
          "Count from 1 to 5.",
          (token: string, done: boolean) => {
            streamCount++;
            if (done) {
              log(`Streaming: ${streamCount} callbacks`, "success");
              resolve();
            }
          },
        );
      });

      // Test 10: Memory tracking (real OS-level data)
      if (memoryTracker) {
        const summary = memoryTracker.getSummary();
        log(
          `Memory tracker: ${summary.snapshotCount} snapshots, buffer ${formatBytes(summary.trackerBufferSizeBytes)}`,
          "success",
        );
        log(
          `RSS: ${formatBytes(summary.currentResidentBytes)} | Peak: ${formatBytes(summary.peakResidentBytes)}`,
          "success",
        );
        log(
          `Native Heap: ${formatBytes(summary.currentNativeHeapBytes)} | Peak: ${formatBytes(summary.peakNativeHeapBytes)}`,
          "success",
        );

        // Also test getMemoryUsage() directly
        const liveUsage = model.getMemoryUsage();
        log(
          `Live RSS: ${formatBytes(liveUsage.residentBytes)} | Native: ${formatBytes(liveUsage.nativeHeapBytes)} | Avail: ${formatBytes(liveUsage.availableMemoryBytes)} | Low: ${liveUsage.isLowMemory}`,
          "success",
        );

        const standalone = createMemoryTracker(10);
        standalone.record({
          timestamp: Date.now(),
          nativeHeapBytes: liveUsage.nativeHeapBytes,
          residentBytes: liveUsage.residentBytes,
          availableMemoryBytes: liveUsage.availableMemoryBytes,
        });
        log(
          `Standalone tracker: ${standalone.getLatestSnapshot() ? "OK" : "FAIL"} (${formatBytes(standalone.getNativeBuffer().byteLength)} native)`,
          standalone.getLatestSnapshot() ? "success" : "error",
        );
      } else {
        log("Memory tracking disabled", "info");
      }

      // Test 11: Image (vision)
      // Vision executor needs ~2GB additional memory on top of the text model.
      // Skip if insufficient memory to avoid jetsam kills.
      const memBeforeImage = model.getMemoryUsage();
      const availGB =
        memBeforeImage.availableMemoryBytes / (1024 * 1024 * 1024);
      if (availGB < 4.0) {
        log(
          `Image skipped: only ${availGB.toFixed(1)} GB available (need ~4 GB for vision encoder)`,
          "info",
        );
      } else {
        try {
          const imagePath = await getTestImagePath(model);
          log(`Image path: ${imagePath}`);
          const imgResp = await model.sendMessageWithImage(
            "Describe this image in one sentence.",
            imagePath,
          );
          log(`Image: "${imgResp}"`, "success");
        } catch (e: any) {
          log(`Image test: ${e.message}`, "error");
        }
      }

      // Test 12: Audio (not supported on iOS yet)
      if (Platform.OS !== "ios") {
        try {
          const audioResp = await model.sendMessageWithAudio(
            "Transcribe this audio.",
            TEST_AUDIO_PATH,
          );
          log(`Audio: "${audioResp}"`, "success");
        } catch (e: any) {
          log(`Audio test: ${e.message}`, "error");
        }
      } else {
        log("Audio skipped (not supported on iOS)", "info");
      }

      log("All tests complete.", "success");
    } catch (err) {
      log(
        `Error: ${err instanceof Error ? err.message : String(err)}`,
        "error",
      );
      console.error(err);
    } finally {
      setIsRunning(false);
    }
  };

  const getStatusLabel = (): { text: string; color: string } => {
    if (error) return { text: "Error", color: THEME.error };
    if (isRunning) return { text: "Running", color: THEME.warning };
    if (isReady) return { text: "Ready", color: THEME.success };
    if (downloadProgress > 0 && downloadProgress < 1)
      return {
        text: `${(downloadProgress * 100).toFixed(0)}%`,
        color: THEME.accent,
      };
    if (downloadProgress === 1)
      return { text: "Loading", color: THEME.warning };
    return { text: "Not Loaded", color: THEME.textDim };
  };

  const status = getStatusLabel();

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Header />

        <View style={styles.grid}>
          <DiagnosticCard
            label="Status"
            value={status.text}
            icon="🔋"
            color={status.color}
          />
          <DiagnosticCard
            label="Latency"
            value={lastLatency ? `${lastLatency}ms` : "--"}
            icon="⏱️"
          />
        </View>

        {tokensPerSec !== null && (
          <View style={[styles.grid, { marginTop: 0 }]}>
            <DiagnosticCard
              label="Speed"
              value={`${tokensPerSec.toFixed(1)} tok/s`}
              icon="⚡"
              color={THEME.success}
            />
          </View>
        )}

        <Section title="Device">
          <InfoRow
            label="Platform"
            value={`${Platform.OS} ${Platform.Version}`}
          />
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Gemma Model</Text>
            <View style={{ flexDirection: "row", gap: 6 }}>
              {(
                Object.keys(MODEL_OPTIONS) as Array<keyof typeof MODEL_OPTIONS>
              ).map((key) => (
                <TouchableOpacity
                  key={key}
                  onPress={() => setSelectedModel(key)}
                  disabled={isReady || isRunning || downloadProgress > 0}
                  style={{
                    paddingHorizontal: 10,
                    paddingVertical: 4,
                    borderRadius: 6,
                    backgroundColor:
                      selectedModel === key ? THEME.accent : THEME.border,
                    opacity:
                      isReady || isRunning || downloadProgress > 0 ? 0.5 : 1,
                  }}
                >
                  <Text style={{ color: THEME.text, fontSize: 12 }}>
                    {key === "gemma3n" ? "3n" : "4"}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Backend</Text>
            <View
              style={{ flexDirection: "row", alignItems: "center", gap: 8 }}
            >
              <Text style={styles.infoValue}>
                {config.backend.toUpperCase()}
              </Text>
              <Switch
                value={useGpu}
                onValueChange={setUseGpu}
                disabled={isReady || isRunning || downloadProgress > 0}
              />
            </View>
          </View>
          <InfoRow
            label="Architecture"
            value={Platform.OS === "ios" ? "arm64" : "arm64"}
          />
        </Section>

        <Section title="Model">
          {!isReady ? (
            <View style={styles.warningBox}>
              <Text style={styles.warningText}>
                {downloadProgress > 0 && downloadProgress < 1
                  ? `Downloading ${(downloadProgress * 100).toFixed(0)}%`
                  : downloadProgress === 1
                    ? "Loading Model..."
                    : "Model Not Loaded"}
              </Text>
              <Text style={styles.instructionText}>
                Download {MODEL_OPTIONS[selectedModel].label} or load from disk.
                {selectedModel === "gemma4" && Platform.OS === "ios"
                  ? "\n⚠️ Gemma 4 requires Extended Virtual Addressing entitlement on iOS."
                  : ""}
              </Text>
              <View style={styles.buttonGroup}>
                <TouchableOpacity
                  style={[
                    styles.primaryButton,
                    downloadProgress > 0 && styles.disabledButton,
                  ]}
                  onPress={load}
                  disabled={downloadProgress > 0 || isRunning}
                >
                  {downloadProgress > 0 ? (
                    <ActivityIndicator color="#fff" />
                  ) : (
                    <Text style={styles.buttonText}>Download & Load</Text>
                  )}
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <>
              <View style={styles.successBox}>
                <Text style={styles.successText}>✓ {MODEL_OPTIONS[selectedModel].label} Ready</Text>
              </View>
              {error && (
                <Text
                  style={{
                    color: THEME.error,
                    fontSize: 13,
                    marginTop: 8,
                  }}
                >
                  {error}
                </Text>
              )}
            </>
          )}
        </Section>

        <Section title="Test Suite">
          <View style={styles.buttonGroup}>
            <TouchableOpacity
              style={[
                styles.primaryButton,
                (!isReady || isRunning) && styles.disabledButton,
              ]}
              onPress={runFullTest}
              disabled={!isReady || isRunning}
            >
              {isRunning ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.buttonText}>Run All Tests</Text>
              )}
            </TouchableOpacity>
            {isReady && (
              <TouchableOpacity
                style={[styles.secondaryButton, { borderColor: THEME.error }]}
                onPress={async () => {
                  try {
                    const fileName = selectedModel === "gemma4"
                      ? "gemma-4-E2B-it.litertlm"
                      : "gemma-3n-E2B-it-int4.litertlm";
                    await deleteModel(fileName);
                    log("Model deleted", "success");
                  } catch (e: any) {
                    log(`Delete failed: ${e.message}`, "error");
                  }
                }}
              >
                <Text style={[styles.buttonText, { color: THEME.error }]}>
                  Delete
                </Text>
              </TouchableOpacity>
            )}
          </View>
        </Section>

        {isReady && (
          <Section title="Quick Chat">
            <View style={styles.chatInputContainer}>
              <TextInput
                style={styles.chatInput}
                placeholder="Type a message..."
                placeholderTextColor={THEME.textDim}
                value={chatInput}
                onChangeText={setChatInput}
                editable={!isRunning}
              />
              <TouchableOpacity
                style={[
                  styles.sendButton,
                  (!chatInput.trim() || isRunning) && styles.disabledButton,
                ]}
                onPress={async () => {
                  if (!model || !chatInput.trim()) return;
                  const msg = chatInput.trim();
                  setChatInput("");
                  setIsRunning(true);
                  log(`You: ${msg}`, "info");
                  try {
                    const t0 = Date.now();
                    const resp = await model.sendMessage(msg);
                    const elapsed = Date.now() - t0;
                    setLastLatency(elapsed);
                    log(`Model: ${resp}`, "success");
                    const stats = model.getStats();
                    setTokensPerSec(stats.tokensPerSecond);
                  } catch (e: any) {
                    log(`Error: ${e.message}`, "error");
                  } finally {
                    setIsRunning(false);
                  }
                }}
                disabled={!chatInput.trim() || isRunning}
              >
                <Text style={styles.buttonText}>Send</Text>
              </TouchableOpacity>
            </View>
          </Section>
        )}

        {memorySummary && memorySummary.snapshotCount > 0 && (
          <Section title="Native Memory Usage">
            <View style={styles.memoryContainer}>
              <Text style={styles.memoryHeader}>Process Memory (Real)</Text>
              <InfoRow
                label="Snapshots"
                value={`${memorySummary.snapshotCount}`}
              />
              <InfoRow
                label="Native Heap"
                value={formatBytes(memorySummary.currentNativeHeapBytes)}
              />
              <InfoRow
                label="Resident (RSS)"
                value={formatBytes(memorySummary.currentResidentBytes)}
              />
              <InfoRow
                label="Average RSS"
                value={formatBytes(memorySummary.averageResidentBytes)}
              />

              <Text style={[styles.memoryHeader, { marginTop: 12 }]}>
                Peak &amp; Delta
              </Text>
              <InfoRow
                label="Peak RSS"
                value={formatBytes(memorySummary.peakResidentBytes)}
              />
              <InfoRow
                label="Peak Native Heap"
                value={formatBytes(memorySummary.peakNativeHeapBytes)}
              />
              <InfoRow
                label="RSS Delta"
                value={`${memorySummary.residentDeltaBytes >= 0 ? "+" : ""}${formatBytes(Math.abs(memorySummary.residentDeltaBytes))}`}
              />

              <View style={styles.memoryTotalRow}>
                <Text style={styles.memoryTotalLabel}>Peak RSS</Text>
                <Text style={styles.memoryTotalValue}>
                  {formatBytes(memorySummary.peakResidentBytes)}
                </Text>
              </View>
            </View>
          </Section>
        )}

        <Section title="System Logs">
          {logs.length === 0 && (
            <Text style={styles.logText}>No logs yet. Run tests to begin.</Text>
          )}
          {logs.map((entry, i) => (
            <Text
              key={i}
              style={[
                styles.logText,
                entry.type === "error" && { color: THEME.error },
                entry.type === "success" && { color: THEME.success },
              ]}
            >
              • {entry.message}
            </Text>
          ))}
        </Section>
      </ScrollView>
    </SafeAreaView>
  );
}

function Header() {
  return (
    <View style={styles.header}>
      <Text style={styles.title}>
        LiteRT <Text style={{ color: THEME.accent }}>LM</Text>
      </Text>
      <Text style={styles.subtitle}>On-Device LLM Inference Engine</Text>
    </View>
  );
}

function DiagnosticCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string;
  icon: string;
  color?: string;
}) {
  return (
    <View style={styles.card}>
      <Text style={styles.cardIcon}>{icon}</Text>
      <Text style={styles.cardLabel}>{label}</Text>
      <Text style={[styles.cardValue, color ? { color } : undefined]}>
        {value}
      </Text>
    </View>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.infoRow}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text style={styles.infoValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: THEME.bg },
  scrollContent: { padding: 20 },
  header: { marginBottom: 24, marginTop: 10 },
  title: {
    fontSize: 32,
    fontWeight: "900",
    color: THEME.text,
    letterSpacing: -1,
  },
  subtitle: { fontSize: 14, color: THEME.textDim, fontWeight: "500" },
  grid: { flexDirection: "row", gap: 12, marginBottom: 20 },
  card: {
    flex: 1,
    backgroundColor: THEME.card,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  cardIcon: { fontSize: 20, marginBottom: 8 },
  cardLabel: {
    fontSize: 12,
    color: THEME.textDim,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  cardValue: {
    fontSize: 20,
    fontWeight: "800",
    color: THEME.text,
    marginTop: 2,
  },
  section: { marginBottom: 24 },
  sectionTitle: {
    fontSize: 13,
    fontWeight: "700",
    color: THEME.textDim,
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
    paddingBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: THEME.border,
  },
  infoLabel: { fontSize: 14, color: THEME.textDim },
  infoValue: { fontSize: 14, color: THEME.text, fontWeight: "600" },
  primaryButton: {
    backgroundColor: THEME.accent,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    minHeight: 48,
    flex: 1,
  },
  secondaryButton: {
    backgroundColor: THEME.card,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    flex: 1,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  buttonGroup: { flexDirection: "row", gap: 12, marginTop: 16 },
  buttonText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 15,
    textAlign: "center",
  },
  disabledButton: { opacity: 0.5 },
  successBox: {
    backgroundColor: "rgba(16, 185, 129, 0.1)",
    padding: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: THEME.success,
  },
  successText: { color: THEME.success, fontWeight: "700", textAlign: "center" },
  warningBox: {
    backgroundColor: "rgba(245, 158, 11, 0.1)",
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: THEME.warning,
    gap: 10,
  },
  warningText: {
    color: THEME.warning,
    fontWeight: "800",
    fontSize: 16,
    textAlign: "center",
  },
  instructionText: {
    color: THEME.textDim,
    fontSize: 13,
    textAlign: "center",
    lineHeight: 18,
  },
  logText: {
    fontSize: 12,
    color: THEME.textDim,
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
    marginBottom: 4,
  },
  memoryContainer: {
    backgroundColor: THEME.card,
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  memoryHeader: {
    fontSize: 12,
    fontWeight: "700",
    color: THEME.accent,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
  memoryTotalRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: THEME.accent,
  },
  memoryTotalLabel: {
    fontSize: 14,
    color: THEME.text,
    fontWeight: "800",
  },
  memoryTotalValue: {
    fontSize: 14,
    color: THEME.accent,
    fontWeight: "800",
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
  },
  chatInputContainer: {
    flexDirection: "row",
    gap: 10,
    alignItems: "center",
  },
  chatInput: {
    flex: 1,
    backgroundColor: THEME.card,
    borderRadius: 12,
    padding: 14,
    color: THEME.text,
    fontSize: 15,
    borderWidth: 1,
    borderColor: THEME.border,
  },
  sendButton: {
    backgroundColor: THEME.accent,
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 12,
    alignItems: "center",
    minHeight: 48,
  },
});
