//
// HybridLiteRTLM.hpp
// react-native-litert-lm
//
// High-performance LLM inference using LiteRT-LM.
// Supports Gemma 4, Gemma 3n, and other .litertlm models.
//
// NOTE: This C++ implementation is used for iOS ONLY.
// Android uses the Kotlin implementation in `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLiteRTLM.kt`.
// Do not assume changes here will affect Android.
//

#pragma once

#include "../nitrogen/generated/shared/c++/HybridLiteRTLMSpec.hpp"

// LiteRT-LM C API (iOS uses prebuilt framework with C ABI)
#ifdef __APPLE__
#include "include/litert_lm_engine.h"
#endif

// Memory usage headers
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#endif
#ifdef __ANDROID__
#include <malloc.h>
#include <fstream>
#endif

#include <string>
#include <optional>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>

namespace margelo::nitro::litertlm {

/**
 * HybridLiteRTLM: React Native bindings for LiteRT-LM.
 * 
 * On iOS, wraps the LiteRT-LM C API (engine.h) with prebuilt framework.
 * On Android, this class is unused — the Kotlin implementation is used instead.
 */
class HybridLiteRTLM : public HybridLiteRTLMSpec {
public:
  HybridLiteRTLM() : HybridObject(TAG) {}
  
  ~HybridLiteRTLM() override {
    close();
  }

  // Prevent copying
  HybridLiteRTLM(const HybridLiteRTLM&) = delete;
  HybridLiteRTLM& operator=(const HybridLiteRTLM&) = delete;

public:
  // HybridLiteRTLMSpec interface implementation
  
  std::shared_ptr<Promise<void>> loadModel(const std::string& modelPath,
                 const std::optional<LLMConfig>& config) override;

  std::shared_ptr<Promise<void>> setTools(const std::string& toolsJson) override;

  std::shared_ptr<Promise<std::string>> sendMessage(const std::string& message) override;
  
  std::shared_ptr<Promise<std::string>> sendMessageWithImage(const std::string& message,
                                   const std::string& imagePath) override;

  std::shared_ptr<Promise<std::string>> downloadModel(const std::string& url, 
                                         const std::string& fileName,
                                         const std::optional<std::function<void(double)>>& onProgress) override;
  
  std::shared_ptr<Promise<void>> deleteModel(const std::string& fileName) override;
  
  std::shared_ptr<Promise<std::string>> sendMessageWithAudio(const std::string& message,
                                   const std::string& audioPath) override;
  
  void sendMessageAsync(
    const std::string& message,
    const std::function<void(const std::string&, bool)>& onToken
  ) override;
  
  std::vector<Message> getHistory() override;
  
  void resetConversation() override;
  
  bool isReady() override;
  
  GenerationStats getStats() override;
  
  MemoryUsage getMemoryUsage() override;
  
  void close() override;

private:
  // LiteRT-LM C API resources (iOS only)
#ifdef __APPLE__
  LiteRtLmEngine* engine_ = nullptr;
  LiteRtLmConversation* conversation_ = nullptr;
  LiteRtLmConversationConfig* conv_config_ = nullptr;
  LiteRtLmSessionConfig* session_config_ = nullptr;
#endif
  
  // State
  bool isLoaded_ = false;
  std::vector<Message> history_;
  GenerationStats lastStats_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  // Thread safety
  mutable std::mutex mutex_;
  
  // Configuration - backend
  Backend backend_ = Backend::CPU;
  
  // System prompt / instruction
  std::string systemPrompt_;

  // Tool definitions (Gemma 4 native function calling). Empty = no tools.
  std::string toolsJson_;

  // Force well-formed tool calls at the token level. Default true when
  // toolsJson_ is non-empty, false otherwise.
  bool enableConstrainedDecoding_ = false;

  // Configuration - sampling parameters
  double temperature_ = 0.7;
  double topK_ = 40.0;
  double topP_ = 0.95;
  double maxTokens_ = 1024.0;

  // Total engine token budget (input + output). Set from LLMConfig.contextWindow,
  // not exposed in JS yet — kept patched at 16384 to fit long tool-laden prompts.
  size_t contextWindow_ = 16384;

  // Helper to ensure model is loaded
  void ensureLoaded() const {
    if (!isLoaded_) {
      throw std::runtime_error("LiteRTLM: No model loaded. Call loadModel() first.");
    }
  }
  
  // Helper to create a new conversation from existing engine
  void createNewConversation();
  
  // JSON helpers for building C API message payloads
  static std::string escapeJson(const std::string& input);
  static std::string buildTextMessageJson(const std::string& text);
  static std::string buildImageMessageJson(const std::string& text, const std::string& imagePath);
  static std::string buildAudioMessageJson(const std::string& text, const std::string& audioPath);
  static std::string extractTextFromResponse(const std::string& jsonResponse);
  
  // Internal implementations (called from Promise lambdas)
  void loadModelInternal(const std::string& modelPath, const std::optional<LLMConfig>& config);
  std::string sendMessageInternal(const std::string& message);
  std::string sendMessageWithImageInternal(const std::string& message, const std::string& imagePath);
  std::string sendMessageWithAudioInternal(const std::string& message, const std::string& audioPath);
  
  // Streaming callback context (must be a plain struct for C function pointer)
  struct StreamContext {
    std::function<void(const std::string&, bool)> onToken;
    std::string fullResponse;
    std::vector<Message>* history;
    std::mutex* historyMutex;
    std::string userMessage;
    GenerationStats* lastStats;
    std::chrono::steady_clock::time_point startTime;
    int tokenCount;
  };
  
  // Static C callback for streaming (no captures needed)
  static void streamCallbackFn(void* callback_data, const char* chunk,
                                bool is_final, const char* error_msg);
};

} // namespace margelo::nitro::litertlm

