//
// HybridLiteRTLM.hpp
// react-native-litert-lm
//
// High-performance LLM inference using LiteRT-LM.
// Supports Gemma 3n and other .litertlm models.
//
// NOTE: This C++ implementation is used for iOS ONLY.
// Android uses the Kotlin implementation in `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLiteRTLM.kt`.
// Do not assume changes here will affect Android.
//

#pragma once

#include "../nitrogen/generated/shared/c++/HybridLiteRTLMSpec.hpp"

// LiteRT-LM headers (conditionally included when available via Prefab/CMake)
#ifdef LITERT_LM_ENABLED
#include "litert/lm/engine.h"
#include "litert/lm/conversation.h"
#include "litert/lm/types.h"
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

namespace margelo::nitro::litertlm {

/**
 * HybridLiteRTLM: React Native bindings for LiteRT-LM.
 * 
 * Wraps LiteRT-LM's Engine and Conversation classes to provide
 * high-level LLM inference with GPU acceleration.
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
  
  void loadModel(const std::string& modelPath, 
                 const std::optional<LLMConfig>& config) override;
  
  std::string sendMessage(const std::string& message) override;
  
  std::string sendMessageWithImage(const std::string& message,
                                   const std::string& imagePath) override;

  std::future<std::string> downloadModel(const std::string& url, 
                                         const std::string& fileName,
                                         const std::optional<std::function<void(double)>>& onProgress) override;
  
  std::string sendMessageWithAudio(const std::string& message,
                                   const std::string& audioPath) override;
  
  void sendMessageAsync(
    const std::string& message,
    const std::function<void(std::string, bool)>& onToken
  ) override;
  
  std::vector<Message> getHistory() override;
  
  void resetConversation() override;
  
  bool isReady() override;
  
  GenerationStats getStats() override;
  
  MemoryUsage getMemoryUsage() override;
  
  void close() override;

private:
  // LiteRT-LM resources (conditionally available on Android with Prefab)
#ifdef LITERT_LM_ENABLED
  std::unique_ptr<litert::lm::Engine> engine_;
  std::unique_ptr<litert::lm::Conversation> conversation_;
#endif
  
  // State
  bool isLoaded_ = false;
  std::vector<Message> history_;
  GenerationStats lastStats_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  
  // Thread safety
  mutable std::mutex mutex_;
  
  // Configuration - backends
  Backend backend_ = Backend::GPU;
  Backend visionBackend_ = Backend::GPU;  // Gemma 3n requires GPU for vision
  Backend audioBackend_ = Backend::CPU;   // Audio typically CPU
  
  // Configuration - sampling parameters
  double temperature_ = 0.7;
  double topK_ = 40.0;
  double topP_ = 0.95;
  double maxTokens_ = 1024.0;
  
  // Helper to ensure model is loaded
  void ensureLoaded() const {
    if (!isLoaded_) {
      throw std::runtime_error("LiteRTLM: No model loaded. Call loadModel() first.");
    }
  }
  
  // Helper to format a message for the engine (apply chat template if needed)
  std::string formatUserPrompt(const std::string& message) const;
  
  // Helper to create a new conversation from existing engine
  void createNewConversation();
};

} // namespace margelo::nitro::litertlm
