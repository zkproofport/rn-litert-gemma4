//
// HybridLiteRTLM.cpp
// react-native-litert-lm
//
// High-performance LLM inference using LiteRT-LM C++ runtime classes
// directly. We previously called the C API (cpp/include/litert_lm_engine.h)
// which exposed only a subset of the engine's capabilities — the C++ API
// gives us channels, overwrite_prompt_template, JsonPreface.extra_context,
// filter_channel_content_from_kv_cache, plus structured tool_calls in the
// response (no more text-fence-block regex parsing).
//
// NOTE: This C++ implementation is used for iOS ONLY.
// Android uses the Kotlin implementation in `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLiteRTLM.kt`.
// Do not assume changes here will affect Android.
//

#include "HybridLiteRTLM.hpp"

#include <NitroModules/Promise.hpp>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <sys/stat.h>
#include <cstdio>

#ifdef __APPLE__
#include "IOSDownloadHelper.h"
#include <os/proc.h>
// Upstream LiteRT-LM C++ runtime headers (resolved via cpp/upstream/* symlinks).
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/file_util.h"
// (HasSpeculativeDecodingSupport from schema/capabilities is built as a
// separate Bazel sublib that's NOT bundled in our prebuilt XCFramework's
// static archive — link fails with "Undefined symbols". We rely on the
// engine itself to handle missing drafter sections gracefully when
// enable_speculative_decoding is set unconditionally.)
#endif
#include <fstream>
#include <thread>
#include <regex>
#include <pthread.h>
#include <functional>

namespace margelo::nitro::litertlm {

// =============================================================================
// Thread Helper — LiteRT engine operations need >512KB stack (XNNPack, Metal)
// =============================================================================

static void runOnLargeStack(std::function<void()> work, size_t stackSize = 8 * 1024 * 1024) {
  struct Context {
    std::function<void()> fn;
    std::exception_ptr exception;
  };
  Context ctx{std::move(work), nullptr};

  pthread_t thread;
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, stackSize);

  int rc = pthread_create(&thread, &attr, [](void* arg) -> void* {
    auto* c = static_cast<Context*>(arg);
    try {
      c->fn();
    } catch (...) {
      c->exception = std::current_exception();
    }
    return nullptr;
  }, &ctx);
  pthread_attr_destroy(&attr);
  if (rc != 0) {
    throw std::runtime_error("Failed to create large-stack thread (errno: " + std::to_string(rc) + ")");
  }
  pthread_join(thread, nullptr);

  if (ctx.exception) {
    std::rethrow_exception(ctx.exception);
  }
}

// =============================================================================
// JSON Helpers
// =============================================================================

std::string HybridLiteRTLM::escapeJson(const std::string& input) {
  std::string output;
  output.reserve(input.size() + 16);
  for (char c : input) {
    switch (c) {
      case '"':  output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      default:   output += c; break;
    }
  }
  return output;
}

std::string HybridLiteRTLM::buildTextMessageJson(const std::string& text) {
  return "{\"role\":\"user\",\"content\":\"" + escapeJson(text) + "\"}";
}

std::string HybridLiteRTLM::buildImageMessageJson(const std::string& text, const std::string& imagePath) {
  return "{\"role\":\"user\",\"content\":["
         "{\"type\":\"text\",\"text\":\"" + escapeJson(text) + "\"},"
         "{\"type\":\"image\",\"path\":\"" + escapeJson(imagePath) + "\"}"
         "]}";
}

std::string HybridLiteRTLM::buildAudioMessageJson(const std::string& text, const std::string& audioPath) {
  return "{\"role\":\"user\",\"content\":["
         "{\"type\":\"text\",\"text\":\"" + escapeJson(text) + "\"},"
         "{\"type\":\"audio\",\"path\":\"" + escapeJson(audioPath) + "\"}"
         "]}";
}

/**
 * Strip Gemma / LiteRT-LM control tokens from model output.
 * The iOS C API returns raw model text including stop/turn markers
 * that the Android Kotlin SDK strips automatically.
 */
static std::string stripControlTokens(const std::string& text) {
  static const char* tokens[] = {
    "<end_of_turn>",
    "<start_of_turn>model",
    "<start_of_turn>user",
    "<start_of_turn>",
    "<eos>",
  };
  std::string result = text;
  for (auto* tok : tokens) {
    std::string t(tok);
    size_t pos;
    while ((pos = result.find(t)) != std::string::npos) {
      result.erase(pos, t.length());
    }
  }
  // Trim leading/trailing whitespace
  size_t start = result.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return "";
  size_t end = result.find_last_not_of(" \t\n\r");
  return result.substr(start, end - start + 1);
}

std::string HybridLiteRTLM::extractTextFromResponse(const std::string& jsonResponse) {
  // The C API response JSON is structured as:
  //   {"role":"model","content":[{"type":"text","text":"..."}]}
  // or:
  //   {"role":"model","content":"..."}
  //
  // We use simple string extraction to avoid a JSON library dependency.
  
  // Try array format first: find "text":"..." after "type":"text"
  std::string textMarker = "\"text\":\"";
  size_t pos = jsonResponse.find("\"type\":\"text\"");
  if (pos != std::string::npos) {
    pos = jsonResponse.find(textMarker, pos);
    if (pos != std::string::npos) {
      pos += textMarker.length();
      std::string result;
      result.reserve(jsonResponse.size() - pos);
      for (size_t i = pos; i < jsonResponse.size(); i++) {
        if (jsonResponse[i] == '\\' && i + 1 < jsonResponse.size()) {
          char next = jsonResponse[i + 1];
          if (next == '"') { result += '"'; i++; }
          else if (next == '\\') { result += '\\'; i++; }
          else if (next == 'n') { result += '\n'; i++; }
          else if (next == 'r') { result += '\r'; i++; }
          else if (next == 't') { result += '\t'; i++; }
          else { result += jsonResponse[i]; }
        } else if (jsonResponse[i] == '"') {
          break;  // End of the text value
        } else {
          result += jsonResponse[i];
        }
      }
      return stripControlTokens(result);
    }
  }
  
  // Try simple string format: "content":"..."
  std::string contentMarker = "\"content\":\"";
  pos = jsonResponse.find(contentMarker);
  if (pos != std::string::npos) {
    pos += contentMarker.length();
    std::string result;
    for (size_t i = pos; i < jsonResponse.size(); i++) {
      if (jsonResponse[i] == '\\' && i + 1 < jsonResponse.size()) {
        char next = jsonResponse[i + 1];
        if (next == '"') { result += '"'; i++; }
        else if (next == '\\') { result += '\\'; i++; }
        else if (next == 'n') { result += '\n'; i++; }
        else { result += jsonResponse[i]; }
      } else if (jsonResponse[i] == '"') {
        break;
      } else {
        result += jsonResponse[i];
      }
    }
    return stripControlTokens(result);
  }
  
  // Fallback: return full response (still strip control tokens)
  return stripControlTokens(jsonResponse);
}

// =============================================================================
// Conversation Management
// =============================================================================

void HybridLiteRTLM::createNewConversation() {
#ifdef __APPLE__
  if (!engine_) {
    throw std::runtime_error("Cannot create conversation: engine not initialized");
  }

  // Reset previous conversation (unique_ptr handles native cleanup).
  conversation_.reset();

  // Build the JsonPreface for the conversation.
  //
  // Messages: optional system prompt as the first message, plus prior turns.
  // Tools: parsed from toolsJson_ (OpenAI-style JSON array — same format the
  // Kotlin SDK passes via its Tool::getToolsDescription()).
  // Extra context: free-form JSON object substituted into the chat template
  // (e.g. {"datetime": "...", "user_location": "Seoul"}).
  //
  // FunctionGemma's chat template treats `content` as either a string OR an
  // array of {type:"text",text:"..."} items. The marker "<<<SECTION>>>"
  // splits the system prompt so the first part renders before tool
  // declarations and the rest renders after — matching Edge Gallery's
  // MobileActions pattern (system intro → tools → datetime).
  ::litert::lm::JsonPreface json_preface;
  nlohmann::ordered_json messages_array = nlohmann::ordered_json::array();

  if (!systemPrompt_.empty()) {
    nlohmann::ordered_json system_msg;
    system_msg["role"] = "system";
    const std::string marker = "<<<SECTION>>>";
    auto pos = systemPrompt_.find(marker);
    if (pos != std::string::npos) {
      auto content_arr = nlohmann::ordered_json::array();
      content_arr.push_back({{"type", "text"},
                             {"text", systemPrompt_.substr(0, pos)}});
      content_arr.push_back({{"type", "text"},
                             {"text", systemPrompt_.substr(pos + marker.size())}});
      system_msg["content"] = std::move(content_arr);
    } else {
      system_msg["content"] = systemPrompt_;
    }
    messages_array.push_back(std::move(system_msg));
  }

  if (!priorMessagesJson_.empty()) {
    try {
      auto prior = nlohmann::ordered_json::parse(priorMessagesJson_);
      if (prior.is_array()) {
        for (auto& m : prior) messages_array.push_back(std::move(m));
      }
    } catch (const std::exception& e) {
      fprintf(stderr, "[FORK] WARNING: priorMessages parse failed: %s\n", e.what());
    }
  }

  json_preface.messages = std::move(messages_array);

  if (!toolsJson_.empty()) {
    try {
      json_preface.tools = nlohmann::ordered_json::parse(toolsJson_);
    } catch (const std::exception& e) {
      fprintf(stderr, "[FORK] WARNING: toolsJson parse failed: %s\n", e.what());
    }
  }

  if (!extraContextJson_.empty()) {
    try {
      json_preface.extra_context = nlohmann::ordered_json::parse(extraContextJson_);
    } catch (const std::exception& e) {
      fprintf(stderr, "[FORK] WARNING: extraContextJson parse failed: %s\n", e.what());
    }
  }

  fprintf(stderr,
    "[FORK] conv_config: tools_set=%d constrained=%d sysPrompt_set=%d "
    "priorMsgs_count=%zu extraCtx_set=%d filterCh=%d\n",
    !toolsJson_.empty() ? 1 : 0,
    enableConstrainedDecoding_ ? 1 : 0,
    !systemPrompt_.empty() ? 1 : 0,
    priorMessagesJson_.empty() ? 0 :
      (nlohmann::ordered_json::accept(priorMessagesJson_) ?
        nlohmann::ordered_json::parse(priorMessagesJson_).size() : 0),
    !extraContextJson_.empty() ? 1 : 0,
    filterChannelContent_ ? 1 : 0);

  auto config_or = ::litert::lm::ConversationConfig::Builder()
    .SetSessionConfig(session_config_)
    .SetPreface(json_preface)
    .SetEnableConstrainedDecoding(enableConstrainedDecoding_)
    .SetFilterChannelContentFromKvCache(filterChannelContent_)
    .Build(*engine_);
  if (!config_or.ok()) {
    fprintf(stderr, "[FORK] ConversationConfig::Build failed: %s\n",
            config_or.status().ToString().c_str());
    throw std::runtime_error("Failed to build conversation config: " +
                             std::string(config_or.status().ToString()));
  }

  auto conv_or = ::litert::lm::Conversation::Create(*engine_, *config_or);
  if (!conv_or.ok()) {
    fprintf(stderr, "[FORK] Conversation::Create failed: %s\n",
            conv_or.status().ToString().c_str());
    throw std::runtime_error("Failed to create conversation: " +
                             std::string(conv_or.status().ToString()));
  }
  conversation_ = std::move(*conv_or);
  fprintf(stderr, "[FORK] conversation created OK conv=%p\n",
          static_cast<void*>(conversation_.get()));
#endif
}

// =============================================================================
// setTools — Replace the active tool list without reloading the model.
// =============================================================================

std::shared_ptr<Promise<void>> HybridLiteRTLM::setTools(const std::string& toolsJson) {
  return Promise<void>::async([this, toolsJson]() {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureLoaded();
    toolsJson_ = toolsJson;
    // Default constrained decoding ON when tools is set.
    enableConstrainedDecoding_ = !toolsJson_.empty();
    history_.clear();
    createNewConversation();
  });
}

// =============================================================================
// loadModel
// =============================================================================

std::shared_ptr<Promise<void>> HybridLiteRTLM::loadModel(
    const std::string& modelPath,
    const std::optional<LLMConfig>& config) {
  return Promise<void>::async([this, modelPath, config]() {
    runOnLargeStack([&]() {
      loadModelInternal(modelPath, config);
    });
  });
}

void HybridLiteRTLM::loadModelInternal(
    const std::string& modelPath,
    const std::optional<LLMConfig>& config) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  if (isLoaded_) {
    close();
  }
  
  if (config.has_value()) {
    if (config->backend.has_value()) {
      backend_ = config->backend.value();
    }
    if (config->temperature.has_value()) {
      temperature_ = config->temperature.value();
    }
    if (config->topK.has_value()) {
      topK_ = config->topK.value();
    }
    if (config->topP.has_value()) {
      topP_ = config->topP.value();
    }
    if (config->maxTokens.has_value()) {
      maxTokens_ = config->maxTokens.value();
    }
    if (config->systemPrompt.has_value()) {
      systemPrompt_ = config->systemPrompt.value();
    }
    if (config->tools.has_value()) {
      toolsJson_ = config->tools.value();
    }
    if (config->enableConstrainedDecoding.has_value()) {
      enableConstrainedDecoding_ = config->enableConstrainedDecoding.value();
    } else {
      // Default: ON when tools is non-empty, OFF otherwise.
      enableConstrainedDecoding_ = !toolsJson_.empty();
    }
    if (config->priorMessages.has_value()) {
      priorMessagesJson_ = config->priorMessages.value();
    }
  }
  
#ifdef __APPLE__
  // Fork debug: redirect stderr to <doc-dir>/litert-stderr.log so the engine's
  // absl::LOG output (which never reaches os_log) can be inspected from the
  // host. Works regardless of whether model is at Documents/<file> (device
  // transfer) or Documents/models/<file> (simulator hardlink).
  {
    static bool stderrRedirected = false;
    if (!stderrRedirected) {
      std::string docDir = modelPath;
      auto lastSlash = docDir.rfind('/');
      if (lastSlash != std::string::npos) docDir = docDir.substr(0, lastSlash);
      // Strip trailing /models if present (so log lands in Documents/)
      const std::string modelsSuffix = "/models";
      if (docDir.size() >= modelsSuffix.size() &&
          docDir.compare(docDir.size() - modelsSuffix.size(), modelsSuffix.size(), modelsSuffix) == 0) {
        docDir = docDir.substr(0, docDir.size() - modelsSuffix.size());
      }
      std::string logPath = docDir + "/litert-stderr.log";
      FILE* tf = fopen(logPath.c_str(), "w"); if (tf) fclose(tf);
      if (freopen(logPath.c_str(), "a", stderr) != nullptr) {
        setvbuf(stderr, nullptr, _IOLBF, 0);
        stderrRedirected = true;
      }
    }
  }
  fprintf(stderr, "[FORK] loadModel: tools_len=%zu constrained=%d\n",
          toolsJson_.size(), enableConstrainedDecoding_ ? 1 : 0);

  // Translate the public Backend enum to the runtime's litert::lm::Backend.
  auto toLitertBackend = [](Backend b) -> ::litert::lm::Backend {
    switch (b) {
      case Backend::GPU: return ::litert::lm::Backend::GPU;
      case Backend::NPU: return ::litert::lm::Backend::GPU;  // iOS: no NPU
      default:           return ::litert::lm::Backend::CPU;
    }
  };

  auto tryCreateEngine = [&](::litert::lm::Backend mainBackend,
                             std::optional<::litert::lm::Backend> visionBackend,
                             std::optional<::litert::lm::Backend> audioBackend) -> bool {
    auto modelAssets = ::litert::lm::ModelAssets::Create(modelPath);
    if (!modelAssets.ok()) {
      fprintf(stderr, "[FORK] ModelAssets::Create failed: %s\n",
              modelAssets.status().ToString().c_str());
      return false;
    }

    auto settings = ::litert::lm::EngineSettings::CreateDefault(
      *std::move(modelAssets), mainBackend, visionBackend, audioBackend);
    if (!settings.ok()) {
      fprintf(stderr, "[FORK] EngineSettings::CreateDefault failed: %s\n",
              settings.status().ToString().c_str());
      return false;
    }

    // contextWindow_ is the total token budget (input + output);
    // maxTokens_ is enforced per-session via session_config below.
    settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      static_cast<int>(contextWindow_));

    // Set cache directory to the same directory as the model file.
    std::string cacheDir = modelPath.substr(0, modelPath.find_last_of('/'));
    settings->GetMutableMainExecutorSettings().SetCacheDir(cacheDir);
    if (visionBackend.has_value() && settings->GetMutableVisionExecutorSettings()) {
      settings->GetMutableVisionExecutorSettings()->SetCacheDir(cacheDir);
    }
    if (audioBackend.has_value() && settings->GetMutableAudioExecutorSettings()) {
      settings->GetMutableAudioExecutorSettings()->SetCacheDir(cacheDir);
    }

    // Speculative decoding (Multi-Token Prediction): up to ~3x decode speedup
    // on GPU per Google's 2026-04-03 Gemma 4 announcement. Edge Gallery
    // wired this in on 2026-05-05 (LlmChatModelHelper.kt commit b5f5993b).
    //
    // Capability detection (HasSpeculativeDecodingSupport) lives in a Bazel
    // sublib (`schema/capabilities`) that our prebuilt XCFramework does NOT
    // include. We instead enable the flag unconditionally and rely on the
    // engine to handle models that lack a drafter section. Empirically:
    // models without a drafter ignore the flag without erroring (verified
    // against Gemma 4 E2B/E4B which DO bundle drafter sections).
    //
    // NOTE: upstream model_allowlists/1_0_13.json declares spec_dec compatible
    // with `llm_chat`/`llm_ask_image`/`llm_ask_audio`/`llm_prompt_lab` —
    // NOT `llm_agent_chat`. We enable across the board to measure impact;
    // if tool-call scenarios break, gate on `enableConstrainedDecoding_ == false`.
    {
      ::litert::lm::AdvancedSettings advanced;
      const auto& existing =
        settings->GetMutableMainExecutorSettings().GetAdvancedSettings();
      if (existing.has_value()) advanced = *existing;
      advanced.enable_speculative_decoding = true;
      settings->GetMutableMainExecutorSettings().SetAdvancedSettings(advanced);
      fprintf(stderr, "[FORK] speculative_decoding=enabled (unconditional, drafter detection at runtime)\n");
    }

    // Enable benchmark info collection (mutates settings to install benchmark
    // params). The accessor return value is intentionally discarded.
    (void)settings->GetMutableBenchmarkParams();

    auto engineOr = ::litert::lm::EngineFactory::CreateDefault(*std::move(settings));
    if (!engineOr.ok()) {
      fprintf(stderr, "[FORK] EngineFactory::CreateDefault failed: %s\n",
              engineOr.status().ToString().c_str());
      return false;
    }
    engine_ = std::move(*engineOr);
    return engine_ != nullptr;
  };

  using ::litert::lm::Backend;
  // Try requested backend first (e.g. GPU/GPU, audio=CPU for multimodal Gemma 4).
  Backend primary = toLitertBackend(backend_);
  const char* requestedName = (primary == Backend::GPU) ? "GPU" : "CPU";
  fprintf(stderr, "[FORK] backend_requested=%s\n", requestedName);
  bool primaryOk = tryCreateEngine(primary, primary, Backend::CPU);
  if (!primaryOk) {
    fprintf(stderr, "[FORK] backend primary=%s FAILED — entering CPU fallback chain\n",
            requestedName);
    // Fallback chain for when the primary backend fails:
    bool fallbackOk = false;
    if (primary != Backend::CPU) {
      // 1) Try CPU main + GPU vision + CPU audio (multimodal models)
      fallbackOk = tryCreateEngine(Backend::CPU, Backend::GPU, Backend::CPU);
      // 2) Try CPU main + CPU vision + CPU audio
      if (!fallbackOk) fallbackOk = tryCreateEngine(Backend::CPU, Backend::CPU, Backend::CPU);
    }
    // 3) Try CPU main + no vision + CPU audio (model has no vision section)
    if (!fallbackOk) fallbackOk = tryCreateEngine(Backend::CPU, std::nullopt, Backend::CPU);
    // 4) Try CPU main + no vision + no audio (text-only model like FunctionGemma 270M)
    if (!fallbackOk) fallbackOk = tryCreateEngine(Backend::CPU, std::nullopt, std::nullopt);
    if (fallbackOk) {
      backend_ = ::margelo::nitro::litertlm::Backend::CPU;
    }
  }
  // Emit a single, easy-to-grep line showing what backend the engine ACTUALLY
  // ended up using. This is what JS-side perf comparisons should be based on.
  if (engine_) {
    const char* resolvedName =
      (backend_ == ::margelo::nitro::litertlm::Backend::GPU) ? "GPU" :
      (backend_ == ::margelo::nitro::litertlm::Backend::NPU) ? "NPU" : "CPU";
    fprintf(stderr, "[FORK] backend_resolved=%s requested=%s primary_ok=%d\n",
            resolvedName, requestedName, primaryOk ? 1 : 0);
  }

  if (!engine_) {
    // Collect diagnostic info.
    std::string diag = " | Diagnostics: ";
    struct stat st;
    if (stat(modelPath.c_str(), &st) == 0) {
      diag += "File size: " + std::to_string(st.st_size) + " bytes";
    } else {
      diag += "Failed to stat file (errno: " + std::to_string(errno) + ")";
    }

    FILE* f = fopen(modelPath.c_str(), "rb");
    if (f) {
      diag += ", Readable: YES";
      fclose(f);
    } else {
      diag += ", Readable: NO (errno: " + std::to_string(errno) + ")";
    }

    throw std::runtime_error(
      "Failed to create LiteRT-LM engine (CPU+GPU paths exhausted). Model path: " +
      modelPath + diag);
  }

  // Build the SessionConfig (kept as a value member so we can re-use it when
  // hot-swapping the conversation in setTools/resetConversation).
  session_config_ = ::litert::lm::SessionConfig::CreateDefault();
  // Sampling parameters via mutable accessor.
  auto& samplerParams = session_config_.GetMutableSamplerParams();
  samplerParams.set_type(::litert::lm::proto::SamplerParameters::TOP_P);
  samplerParams.set_k(static_cast<int>(topK_));
  samplerParams.set_p(static_cast<float>(topP_));
  samplerParams.set_temperature(static_cast<float>(temperature_));
  samplerParams.set_seed(0);
  // max_output_tokens is set per-session.
  session_config_.SetMaxOutputTokens(static_cast<int>(maxTokens_));

  if (engine_->GetEngineSettings().GetVisionExecutorSettings().has_value()) {
    session_config_.SetVisionModalityEnabled(true);
  }
  if (engine_->GetEngineSettings().GetAudioExecutorSettings().has_value()) {
    session_config_.SetAudioModalityEnabled(true);
  }

  createNewConversation();
#endif
  
  isLoaded_ = true;
  history_.clear();
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

// =============================================================================
// sendMessage — Blocking text inference
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::sendMessage(const std::string& message) {
  return Promise<std::string>::async([this, message]() -> std::string {
    std::string result;
    runOnLargeStack([&]() {
      result = sendMessageInternal(message);
    });
    return result;
  });
}

std::string HybridLiteRTLM::sendMessageInternal(const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::steady_clock::now();
  std::string result;
  
#ifdef __APPLE__
  std::string msgJson = buildTextMessageJson(message);
  fprintf(stderr, "[FORK] sendMessage: msgJson_len=%zu conv=%p\n",
          msgJson.size(), static_cast<void*>(conversation_.get()));

  ::litert::lm::Message userMsg = nlohmann::ordered_json::parse(msgJson);
  auto responseOr = conversation_->SendMessage(userMsg, ::litert::lm::OptionalArgs());
  if (!responseOr.ok()) {
    fprintf(stderr, "[FORK] sendMessage failed: %s\n",
            responseOr.status().ToString().c_str());
    throw std::runtime_error("LiteRT-LM: sendMessage failed: " +
                             std::string(responseOr.status().ToString()));
  }
  std::string raw = responseOr->dump();
  fprintf(stderr, "[FORK] raw response (%zu bytes): %s\n",
          raw.size(),
          raw.size() > 500 ? (raw.substr(0, 500) + "...").c_str() : raw.c_str());
  result = extractTextFromResponse(raw);

  auto benchInfoOr = conversation_->GetBenchmarkInfo();
  if (benchInfoOr.ok()) {
    const auto& benchInfo = *benchInfoOr;
    int numDecodeTurns = benchInfo.GetTotalDecodeTurns();
    if (numDecodeTurns > 0) {
      int lastIdx = numDecodeTurns - 1;
      lastStats_.tokensPerSecond = benchInfo.GetDecodeTokensPerSec(lastIdx);
      auto turn = benchInfo.GetDecodeTurn(lastIdx);
      if (turn.ok()) {
        lastStats_.completionTokens = static_cast<double>(turn->num_tokens);
      }
    }
    lastStats_.timeToFirstToken = benchInfo.GetTimeToFirstToken();
  }
#else
  // Non-Apple stub
  result = "[iOS only] LiteRT-LM inference not available on this platform.";
#endif
  
  auto endTime = std::chrono::steady_clock::now();
  double latencyMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
  lastStats_.totalTime = latencyMs / 1000.0;
  
  // Update history
  history_.push_back(Message{Role::USER, message});
  history_.push_back(Message{Role::MODEL, result});
  
  return result;
}

// =============================================================================
// sendMessageAsync — Streaming text inference
// =============================================================================

// streamCallbackFn (legacy C-pointer callback) is no longer used — the C++
// SendMessageAsync takes an absl::AnyInvocable lambda directly. Kept declared
// in HybridLiteRTLM.hpp as a no-op stub to preserve ABI in case Nitro
// regenerates a header that references it; safe to remove later.
void HybridLiteRTLM::streamCallbackFn(void*, const char*, bool, const char*) {}

void HybridLiteRTLM::sendMessageAsync(
    const std::string& message,
    const std::function<void(const std::string&, bool)>& onToken) {

  auto onTokenCopy = onToken;
  auto messageCopy = message;

#ifdef __APPLE__
  ensureLoaded();

  std::string msgJson = buildTextMessageJson(messageCopy);
  ::litert::lm::Message userMsg = nlohmann::ordered_json::parse(msgJson);

  // Shared state for the streaming callback. Captured by lambda copy.
  struct StreamState {
    std::function<void(const std::string&, bool)> onToken;
    std::string fullResponse;
    std::vector<Message>* history;
    std::mutex* historyMutex;
    std::string userMessage;
    GenerationStats* lastStats;
    std::chrono::steady_clock::time_point startTime;
    int tokenCount;
  };
  auto state = std::make_shared<StreamState>();
  state->onToken = std::move(onTokenCopy);
  state->history = &history_;
  state->historyMutex = &mutex_;
  state->userMessage = messageCopy;
  state->lastStats = &lastStats_;
  state->startTime = std::chrono::steady_clock::now();
  state->tokenCount = 0;

  runOnLargeStack([&]() {
    auto status = conversation_->SendMessageAsync(
      userMsg,
      [state](absl::StatusOr<::litert::lm::Message> chunkOr) mutable {
        if (!chunkOr.ok()) {
          state->onToken(std::string("Error: ") +
                         std::string(chunkOr.status().message()), true);
          return;
        }
        const ::litert::lm::Message& chunkMsg = *chunkOr;
        if (chunkMsg.empty()) {
          // End-of-stream sentinel.
          auto endTime = std::chrono::steady_clock::now();
          double durationMs = std::chrono::duration<double, std::milli>(
            endTime - state->startTime).count();
          if (state->lastStats && state->tokenCount > 0) {
            state->lastStats->completionTokens =
              static_cast<double>(state->tokenCount);
            state->lastStats->totalTime = durationMs / 1000.0;
            state->lastStats->tokensPerSecond =
              (state->tokenCount / durationMs) * 1000.0;
          }
          {
            std::lock_guard<std::mutex> lock(*state->historyMutex);
            state->history->push_back(Message{Role::USER, state->userMessage});
            state->history->push_back(Message{Role::MODEL, state->fullResponse});
          }
          state->onToken("", true);
          return;
        }
        // Extract text from the structured chunk Message JSON. The engine may
        // return content as a string OR as an array of {type:"text",text:"..."}
        // entries; both are handled by extractTextFromResponse.
        std::string cleaned = extractTextFromResponse(chunkMsg.dump());
        if (!cleaned.empty()) {
          state->fullResponse += cleaned;
          state->tokenCount++;
          state->onToken(cleaned, false);
        }
      },
      ::litert::lm::OptionalArgs());

    if (!status.ok()) {
      throw std::runtime_error("LiteRT-LM: Failed to start streaming: " +
                               std::string(status.ToString()));
    }
  });
#else
  // Non-Apple stub
  onTokenCopy("[iOS only] Streaming not available on this platform.", true);
#endif
}

// =============================================================================
// sendMessageWithImage — Multimodal (vision)
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::sendMessageWithImage(
    const std::string& message,
    const std::string& imagePath) {
  return Promise<std::string>::async([this, message, imagePath]() -> std::string {
    std::string result;
    runOnLargeStack([&]() {
      result = sendMessageWithImageInternal(message, imagePath);
    });
    return result;
  });
}

std::string HybridLiteRTLM::sendMessageWithImageInternal(
    const std::string& message,
    const std::string& imagePath) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::steady_clock::now();
  std::string result;
  
#ifdef __APPLE__
  std::ifstream imageFile(imagePath);
  if (!imageFile.good()) {
    throw std::runtime_error("Image file not found: " + imagePath);
  }
  imageFile.close();

  std::string msgJson = buildImageMessageJson(message, imagePath);
  ::litert::lm::Message userMsg = nlohmann::ordered_json::parse(msgJson);
  auto responseOr = conversation_->SendMessage(userMsg, ::litert::lm::OptionalArgs());
  if (!responseOr.ok()) {
    throw std::runtime_error("LiteRT-LM: sendMessageWithImage failed: " +
                             std::string(responseOr.status().ToString()));
  }
  std::string raw = responseOr->dump();
  fprintf(stderr, "[FORK] raw response (%zu bytes): %s\n",
          raw.size(),
          raw.size() > 500 ? (raw.substr(0, 500) + "...").c_str() : raw.c_str());
  result = extractTextFromResponse(raw);
#else
  result = "[iOS only] Vision inference not available on this platform.";
#endif
  
  auto endTime = std::chrono::steady_clock::now();
  lastStats_.totalTime = std::chrono::duration<double>(endTime - startTime).count();
  
  history_.push_back(Message{Role::USER, message + " [image: " + imagePath + "]"});
  history_.push_back(Message{Role::MODEL, result});
  
  return result;
}

// =============================================================================
// sendMessageWithAudio — Multimodal (audio)
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::sendMessageWithAudio(
    const std::string& message,
    const std::string& audioPath) {
  return Promise<std::string>::async([this, message, audioPath]() -> std::string {
    std::string result;
    runOnLargeStack([&]() {
      result = sendMessageWithAudioInternal(message, audioPath);
    });
    return result;
  });
}

std::string HybridLiteRTLM::sendMessageWithAudioInternal(
    const std::string& message,
    const std::string& audioPath) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::steady_clock::now();
  std::string result;
  
#ifdef __APPLE__
  std::ifstream audioFile(audioPath);
  if (!audioFile.good()) {
    throw std::runtime_error("Audio file not found: " + audioPath);
  }
  audioFile.close();

  std::string msgJson = buildAudioMessageJson(message, audioPath);
  ::litert::lm::Message userMsg = nlohmann::ordered_json::parse(msgJson);
  auto responseOr = conversation_->SendMessage(userMsg, ::litert::lm::OptionalArgs());
  if (!responseOr.ok()) {
    throw std::runtime_error("LiteRT-LM: sendMessageWithAudio failed: " +
                             std::string(responseOr.status().ToString()));
  }
  std::string raw = responseOr->dump();
  fprintf(stderr, "[FORK] raw response (%zu bytes): %s\n",
          raw.size(),
          raw.size() > 500 ? (raw.substr(0, 500) + "...").c_str() : raw.c_str());
  result = extractTextFromResponse(raw);
#else
  result = "[iOS only] Audio inference not available on this platform.";
#endif
  
  auto endTime = std::chrono::steady_clock::now();
  lastStats_.totalTime = std::chrono::duration<double>(endTime - startTime).count();
  
  history_.push_back(Message{Role::USER, message + " [audio: " + audioPath + "]"});
  history_.push_back(Message{Role::MODEL, result});
  
  return result;
}

// =============================================================================
// downloadModel — Download model from URL
// =============================================================================

std::shared_ptr<Promise<std::string>> HybridLiteRTLM::downloadModel(
    const std::string& url,
    const std::string& fileName,
    const std::optional<std::function<void(double)>>& onProgress) {
  return Promise<std::string>::async([url, fileName, onProgress]() -> std::string {
#ifdef __APPLE__
    return litert_lm::downloadModelFile(url, fileName, onProgress);
#else
    // Non-Apple platforms: not supported from C++ (Android uses Kotlin)
    throw std::runtime_error("Download not available on this platform. Use the Kotlin implementation.");
#endif
  });
}

std::shared_ptr<Promise<void>> HybridLiteRTLM::deleteModel(const std::string& fileName) {
  return Promise<void>::async([fileName]() {
    std::string path;
#ifdef __APPLE__
    // Match the path used by IOSDownloadHelper: ~/Library/Caches/litert_models/
    const char* home = getenv("HOME");
    if (home) {
      path = std::string(home) + "/Library/Caches/litert_models/" + fileName;
    }
#else
    path = "/tmp/" + fileName;
#endif
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  });
}

// =============================================================================
// getHistory
// =============================================================================

std::vector<Message> HybridLiteRTLM::getHistory() {
  std::lock_guard<std::mutex> lock(mutex_);
  return history_;
}

// =============================================================================
// resetConversation
// =============================================================================

void HybridLiteRTLM::resetConversation() {
  std::lock_guard<std::mutex> lock(mutex_);
  
  history_.clear();
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
#ifdef __APPLE__
  if (isLoaded_ && engine_) {
    createNewConversation();
  }
#endif
}

// =============================================================================
// isReady
// =============================================================================

bool HybridLiteRTLM::isReady() {
  std::lock_guard<std::mutex> lock(mutex_);
  return isLoaded_;
}

// =============================================================================
// getStats
// =============================================================================

GenerationStats HybridLiteRTLM::getStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastStats_;
}

// =============================================================================
// getMemoryUsage — Uses Mach APIs for iOS process memory
// =============================================================================

MemoryUsage HybridLiteRTLM::getMemoryUsage() {
  double nativeHeapBytes = 0;
  double residentBytes = 0;
  double availableBytes = 0;
  bool isLowMemory = false;
  
#ifdef __APPLE__
  // Get app process memory (resident set size)
  struct mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  
  kern_return_t kr = task_info(mach_task_self(),
                               MACH_TASK_BASIC_INFO,
                               (task_info_t)&info,
                               &count);
  
  if (kr == KERN_SUCCESS) {
    residentBytes = static_cast<double>(info.resident_size);
    // On iOS, mach_task_basic_info doesn't separate heap from RSS.
    // Use resident_size_max as a proxy for peak native allocation.
    nativeHeapBytes = static_cast<double>(info.resident_size);
  }
  
  // Use os_proc_available_memory() (iOS 13+) for accurate Jetsam headroom.
  // This reports how much memory the process can still allocate before
  // the system kills it — far more accurate than total_physical - process_rss.
  availableBytes = static_cast<double>(os_proc_available_memory());
  
  // Low memory threshold (~200MB available)
  isLowMemory = availableBytes < 200.0 * 1024.0 * 1024.0;
#endif
  
  return MemoryUsage{
    nativeHeapBytes,            // nativeHeapBytes (RSS as proxy on iOS)
    residentBytes,              // residentBytes  
    availableBytes,             // availableMemoryBytes
    isLowMemory                 // isLowMemory
  };
}

// =============================================================================
// close — Clean up all LiteRT-LM resources
// =============================================================================

void HybridLiteRTLM::close() {
  // Note: Don't lock here if called from destructor (mutex may be destroyed)
  // The caller (loadModel, destructor) should handle locking.
  
  isLoaded_ = false;
  history_.clear();
  
#ifdef __APPLE__
  // unique_ptrs handle native cleanup. Order: conversation first so it stops
  // referencing the engine, then engine. session_config_ is a value member —
  // reassigning to a fresh default releases any held state.
  conversation_.reset();
  engine_.reset();
  session_config_ = ::litert::lm::SessionConfig::CreateDefault();
#endif
  
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

} // namespace margelo::nitro::litertlm
