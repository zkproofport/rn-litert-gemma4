//
// HybridLiteRTLM.cpp
// react-native-litert-lm
//
// High-performance LLM inference using LiteRT-LM C API.
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
      return result;
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
    return result;
  }
  
  // Fallback: return full response
  return jsonResponse;
}

// =============================================================================
// Conversation Management
// =============================================================================

void HybridLiteRTLM::createNewConversation() {
#ifdef __APPLE__
  if (!engine_) {
    throw std::runtime_error("Cannot create conversation: engine not initialized");
  }
  
  // Clean up previous conversation
  if (conversation_) {
    litert_lm_conversation_delete(conversation_);
    conversation_ = nullptr;
  }
  if (conv_config_) {
    litert_lm_conversation_config_delete(conv_config_);
    conv_config_ = nullptr;
  }
  
  // Build system message JSON if provided
  std::string systemMsgJson;
  const char* systemMsgPtr = nullptr;
  if (!systemPrompt_.empty()) {
    systemMsgJson = "{\"role\":\"system\",\"content\":\"" + escapeJson(systemPrompt_) + "\"}";
    systemMsgPtr = systemMsgJson.c_str();
  }
  
  // Create conversation config with session config
  conv_config_ = litert_lm_conversation_config_create(
    engine_,
    session_config_,  // may be nullptr for defaults
    systemMsgPtr,     // system message
    nullptr,          // tools (not used yet)
    nullptr,          // messages history
    false             // constrained decoding
  );
  if (!conv_config_) {
    throw std::runtime_error("Failed to create conversation config");
  }
  
  // Create conversation
  conversation_ = litert_lm_conversation_create(engine_, conv_config_);
  if (!conversation_) {
    litert_lm_conversation_config_delete(conv_config_);
    conv_config_ = nullptr;
    throw std::runtime_error("Failed to create conversation");
  }
#endif
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
  }
  
#ifdef __APPLE__
  // Set log verbosity: 2=WARNING (production), 0=INFO (debug)
  litert_lm_set_min_log_level(2);

  auto backendStr = [](Backend b) -> const char* {
    switch (b) {
      case Backend::GPU: return "gpu";
      case Backend::NPU: return "gpu"; // NPU not available on iOS, use GPU
      default: return "cpu";
    }
  };
  
  auto tryCreateEngine = [&](const char* backend, const char* visionBackend) -> bool {
    auto* settings = litert_lm_engine_settings_create(
      modelPath.c_str(),
      backend,
      visionBackend,
      nullptr // audio executor not supported on iOS yet
    );
    if (!settings) {
      return false;
    }
    
    litert_lm_engine_settings_set_max_num_tokens(settings, static_cast<int>(maxTokens_));
    litert_lm_engine_settings_enable_benchmark(settings);
    
    // Set cache directory to the same directory as the model file
    std::string cacheDir = modelPath.substr(0, modelPath.find_last_of('/'));
    litert_lm_engine_settings_set_cache_dir(settings, cacheDir.c_str());
    
    engine_ = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    
    return engine_ != nullptr;
  };
  
  // Try requested backend first (e.g. gpu/gpu)
  const char* primaryBackend = backendStr(backend_);
  if (!tryCreateEngine(primaryBackend, primaryBackend)) {
    // Fallback chain for when the primary backend fails:
    bool fallbackOk = false;
    if (backend_ != Backend::CPU) {
      // 1) Try CPU main + GPU vision (model's vision encoder often requires GPU)
      fallbackOk = tryCreateEngine("cpu", "gpu");
      // 2) Try CPU main + CPU vision
      if (!fallbackOk) fallbackOk = tryCreateEngine("cpu", "cpu");
    }
    // 3) Try CPU main + no vision (nullptr skips vision executor entirely)
    if (!fallbackOk) fallbackOk = tryCreateEngine("cpu", nullptr);
    if (fallbackOk) {
      backend_ = Backend::CPU;
    }
  }
  
  if (!engine_) {
    // Collect diagnostic info
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
    
    // Get the native error from the C API
    const char* nativeErr = litert_lm_get_last_error();
    if (nativeErr && nativeErr[0] != '\0') {
      diag += " | Native error: " + std::string(nativeErr);
    }

    throw std::runtime_error(
      "Failed to create LiteRT-LM engine. Tried backend '" +
      std::string(primaryBackend) + "' and CPU fallback. Model path: " + modelPath + diag);
  }
  
  session_config_ = litert_lm_session_config_create();
  if (session_config_) {
    litert_lm_session_config_set_max_output_tokens(session_config_, static_cast<int>(maxTokens_));
    
    LiteRtLmSamplerParams sampler{};
    sampler.type = kTopP;
    sampler.top_k = static_cast<int32_t>(topK_);
    sampler.top_p = static_cast<float>(topP_);
    sampler.temperature = static_cast<float>(temperature_);
    sampler.seed = 0;
    litert_lm_session_config_set_sampler_params(session_config_, &sampler);
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
  
  auto* response = litert_lm_conversation_send_message(
    conversation_, msgJson.c_str(), nullptr);
  
  if (!response) {
    throw std::runtime_error("LiteRT-LM: sendMessage failed");
  }
  
  const char* responseStr = litert_lm_json_response_get_string(response);
  if (responseStr) {
    result = extractTextFromResponse(std::string(responseStr));
  }
  litert_lm_json_response_delete(response);
  
  auto* benchInfo = litert_lm_conversation_get_benchmark_info(conversation_);
  if (benchInfo) {
    int numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchInfo);
    if (numDecodeTurns > 0) {
      int lastIdx = numDecodeTurns - 1;
      lastStats_.tokensPerSecond = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(benchInfo, lastIdx);
      lastStats_.completionTokens = static_cast<double>(
        litert_lm_benchmark_info_get_decode_token_count_at(benchInfo, lastIdx));
    }
    lastStats_.timeToFirstToken = litert_lm_benchmark_info_get_time_to_first_token(benchInfo);
    litert_lm_benchmark_info_delete(benchInfo);
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

void HybridLiteRTLM::streamCallbackFn(void* callback_data, const char* chunk,
                                        bool is_final, const char* error_msg) {
  auto* ctx = static_cast<StreamContext*>(callback_data);
  
  if (error_msg) {
    // Error occurred — notify JS and clean up
    ctx->onToken(std::string("Error: ") + error_msg, true);
    delete ctx;
    return;
  }
  
  if (is_final) {
    // Calculate stats
    auto endTime = std::chrono::steady_clock::now();
    double durationMs = std::chrono::duration<double, std::milli>(endTime - ctx->startTime).count();
    
    if (ctx->lastStats && ctx->tokenCount > 0) {
      ctx->lastStats->completionTokens = static_cast<double>(ctx->tokenCount);
      ctx->lastStats->totalTime = durationMs / 1000.0;
      ctx->lastStats->tokensPerSecond = (ctx->tokenCount / durationMs) * 1000.0;
    }
    
    // Update history (thread-safe)
    {
      std::lock_guard<std::mutex> lock(*ctx->historyMutex);
      ctx->history->push_back(Message{Role::USER, ctx->userMessage});
      ctx->history->push_back(Message{Role::MODEL, ctx->fullResponse});
    }
    
    ctx->onToken("", true);
    delete ctx;
    return;
  }
  
  if (chunk) {
    std::string token(chunk);
    ctx->fullResponse += token;
    ctx->tokenCount++;
    ctx->onToken(token, false);
  }
}

void HybridLiteRTLM::sendMessageAsync(
    const std::string& message,
    const std::function<void(const std::string&, bool)>& onToken) {
  
  // Copy values for the background thread (avoid use-after-free)
  auto onTokenCopy = onToken;
  auto messageCopy = message;
  
  // Capture shared state safely — use unique_ptr to prevent leaks
  auto ctxOwner = std::make_unique<StreamContext>();
  ctxOwner->onToken = std::move(onTokenCopy);
  ctxOwner->fullResponse = "";
  ctxOwner->history = &history_;
  ctxOwner->historyMutex = &mutex_;
  ctxOwner->userMessage = messageCopy;
  ctxOwner->lastStats = &lastStats_;
  ctxOwner->startTime = std::chrono::steady_clock::now();
  ctxOwner->tokenCount = 0;
  
#ifdef __APPLE__
  ensureLoaded();
  
  std::string msgJson = buildTextMessageJson(messageCopy);
  
  // Release ownership — the C callback now owns the context via raw pointer.
  // streamCallbackFn will delete it when done or on error.
  StreamContext* ctx = ctxOwner.release();
  
  // Wrap the initial engine call in runOnLargeStack for consistency
  // with all other engine entry points (XNNPack needs >512KB stack).
  runOnLargeStack([&]() {
    int result = litert_lm_conversation_send_message_stream(
      conversation_, msgJson.c_str(), nullptr,
      streamCallbackFn, ctx);
    
    if (result != 0) {
      delete ctx;
      throw std::runtime_error("LiteRT-LM: Failed to start streaming inference");
    }
  });
#else
  // Non-Apple stub
  ctxOwner->onToken("[iOS only] Streaming not available on this platform.", true);
  // ctxOwner auto-deleted by unique_ptr
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
  // Verify image exists
  std::ifstream imageFile(imagePath);
  if (!imageFile.good()) {
    throw std::runtime_error("Image file not found: " + imagePath);
  }
  imageFile.close();
  
  // Build multimodal message JSON — the C API handles image preprocessing
  std::string msgJson = buildImageMessageJson(message, imagePath);
  
  auto* response = litert_lm_conversation_send_message(
    conversation_, msgJson.c_str(), nullptr);
  
  if (!response) {
    std::string errMsg = "LiteRT-LM: sendMessageWithImage failed";
    const char* nativeErr = litert_lm_get_last_error();
    if (nativeErr && nativeErr[0] != '\0') {
      errMsg += ": " + std::string(nativeErr);
    }
    throw std::runtime_error(errMsg);
  }
  
  const char* responseStr = litert_lm_json_response_get_string(response);
  if (responseStr) {
    result = extractTextFromResponse(std::string(responseStr));
  }
  litert_lm_json_response_delete(response);
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
  
  auto* response = litert_lm_conversation_send_message(
    conversation_, msgJson.c_str(), nullptr);
  
  if (!response) {
    throw std::runtime_error("LiteRT-LM: sendMessageWithAudio failed");
  }
  
  const char* responseStr = litert_lm_json_response_get_string(response);
  if (responseStr) {
    result = extractTextFromResponse(std::string(responseStr));
  }
  litert_lm_json_response_delete(response);
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
  if (conversation_) {
    litert_lm_conversation_delete(conversation_);
    conversation_ = nullptr;
  }
  if (conv_config_) {
    litert_lm_conversation_config_delete(conv_config_);
    conv_config_ = nullptr;
  }
  if (session_config_) {
    litert_lm_session_config_delete(session_config_);
    session_config_ = nullptr;
  }
  if (engine_) {
    litert_lm_engine_delete(engine_);
    engine_ = nullptr;
  }
#endif
  
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

} // namespace margelo::nitro::litertlm
