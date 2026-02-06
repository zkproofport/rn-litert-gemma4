//
// HybridLiteRTLM.cpp
// react-native-litert-lm
//
// High-performance LLM inference using LiteRT-LM.
//
// NOTE: This C++ implementation is used for iOS ONLY.
// Android uses the Kotlin implementation in `android/src/main/java/com/margelo/nitro/dev/litert/litertlm/HybridLiteRTLM.kt`.
// Do not assume changes here will affect Android.
//

#include "HybridLiteRTLM.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#include <chrono>
#include <stdexcept>
#include <sstream>
#include <fstream>

namespace margelo::nitro::litertlm {

//------------------------------------------------------------------------------
// Helper: Format user prompt (applies chat template if needed)
//------------------------------------------------------------------------------
std::string HybridLiteRTLM::formatUserPrompt(const std::string& message) const {
  // The LiteRT-LM Conversation class handles chat templates internally,
  // so we just return the message as-is. If we were using Session directly,
  // we'd apply the Gemma/Phi template here.
  return message;
}

//------------------------------------------------------------------------------
// Helper: Create a new Conversation from existing Engine
//------------------------------------------------------------------------------
void HybridLiteRTLM::createNewConversation() {
#ifdef LITERT_LM_ENABLED
  if (!engine_) {
    throw std::runtime_error("Cannot create conversation: engine not initialized");
  }
  
  auto conversation_config = litert::lm::ConversationConfig::CreateDefault(*engine_);
  if (!conversation_config.ok()) {
    throw std::runtime_error("Failed to create conversation config: " + 
        std::string(conversation_config.status().message()));
  }
  
  auto conversation = litert::lm::Conversation::Create(*engine_, *conversation_config);
  if (!conversation.ok()) {
    throw std::runtime_error("Failed to create conversation: " + 
        std::string(conversation.status().message()));
  }
  conversation_ = std::move(*conversation);
#endif
}

//------------------------------------------------------------------------------
// loadModel - Initialize Engine and Conversation
//------------------------------------------------------------------------------
void HybridLiteRTLM::loadModel(
    const std::string& modelPath,
    const std::optional<LLMConfig>& config) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Clean up existing resources
  if (isLoaded_) {
    isLoaded_ = false;
    history_.clear();
#ifdef LITERT_LM_ENABLED
    conversation_.reset();
    engine_.reset();
#endif
  }
  
  // Apply configuration
  if (config.has_value()) {
    if (config->backend.has_value()) {
      backend_ = config->backend.value();
    }
    if (config->visionBackend.has_value()) {
      visionBackend_ = config->visionBackend.value();
    }
    if (config->audioBackend.has_value()) {
      audioBackend_ = config->audioBackend.value();
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
  }
  
#ifdef LITERT_LM_ENABLED
  // 1. Create ModelAssets from model path
  auto model_assets = litert::lm::ModelAssets::Create(modelPath);
  if (!model_assets.ok()) {
    throw std::runtime_error("Failed to load model assets: " + 
        std::string(model_assets.status().message()));
  }

  // 2. Map our Backend enum to LiteRT-LM Backend enum
  auto engine_backend = (backend_ == Backend::GPU) 
    ? litert::lm::Backend::GPU 
    : litert::lm::Backend::CPU;
  auto vision_backend = (visionBackend_ == Backend::GPU)
    ? litert::lm::Backend::GPU 
    : litert::lm::Backend::CPU;
  auto audio_backend = (audioBackend_ == Backend::GPU)
    ? litert::lm::Backend::GPU 
    : litert::lm::Backend::CPU;

  // 3. Create EngineSettings with all backends
  auto engine_settings = litert::lm::EngineSettings::CreateDefault(
    *model_assets, 
    engine_backend,
    vision_backend,
    audio_backend
  );

  // 4. Create the Engine (heavyweight - loads model weights)
  auto engine = litert::lm::Engine::CreateEngine(engine_settings);
  if (!engine.ok()) {
    throw std::runtime_error("Failed to create engine: " + 
        std::string(engine.status().message()));
  }
  engine_ = std::move(*engine);

  // 5. Create the Conversation (lightweight - holds KV cache)
  createNewConversation();
  
#endif // LITERT_LM_ENABLED
  
  isLoaded_ = true;
  history_.clear();
  
  // Reset stats
  lastStats_ = GenerationStats{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

//------------------------------------------------------------------------------
// sendMessage - Blocking text inference
//------------------------------------------------------------------------------
std::string HybridLiteRTLM::sendMessage(const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
  auto startTime = std::chrono::high_resolution_clock::now();
  
  // Add user message to history
  Message userMessage;
  userMessage.role = Role::USER;
  userMessage.content = message;
  history_.push_back(userMessage);
  
  std::string responseText;
  
#ifdef LITERT_LM_ENABLED
  // Build the message struct for LiteRT-LM
  // The Conversation API expects a structured input
  litert::lm::UserMessage lm_message;
  lm_message.role = "user";
  lm_message.content = message;
  
  auto response = conversation_->SendMessage(lm_message);
  if (!response.ok()) {
    // Remove the user message we just added since inference failed
    history_.pop_back();
    throw std::runtime_error("Inference failed: " + 
        std::string(response.status().message()));
  }
  
  responseText = response->content;
  
  // Update stats from response if available
  if (response->stats.has_value()) {
    const auto& stats = response->stats.value();
    lastStats_.promptTokens = static_cast<double>(stats.prompt_tokens);
    lastStats_.completionTokens = static_cast<double>(stats.completion_tokens);
    lastStats_.totalTokens = lastStats_.promptTokens + lastStats_.completionTokens;
    lastStats_.timeToFirstToken = stats.time_to_first_token_ms;
    lastStats_.totalTime = stats.total_time_ms;
    lastStats_.tokensPerSecond = (lastStats_.totalTime > 0) 
      ? lastStats_.completionTokens / (lastStats_.totalTime / 1000.0)
      : 0.0;
  }
  
#else
  // Stub response when LiteRT-LM is not available
  responseText = "[LiteRT-LM Stub] Model response placeholder. "
    "Real inference will be available when LiteRT-LM libraries are integrated. "
    "You said: " + message;
  
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
  
  // Estimate stats for stub
  lastStats_.promptTokens = static_cast<double>(message.length() / 4);
  lastStats_.completionTokens = static_cast<double>(responseText.length() / 4);
  lastStats_.totalTokens = lastStats_.promptTokens + lastStats_.completionTokens;
  lastStats_.totalTime = static_cast<double>(duration);
  lastStats_.timeToFirstToken = lastStats_.totalTime / 2;
  lastStats_.tokensPerSecond = (lastStats_.totalTime > 0) 
    ? lastStats_.completionTokens / (lastStats_.totalTime / 1000.0)
    : 0;
#endif
  
  // Add model response to history
  Message modelMessage;
  modelMessage.role = Role::MODEL;
  modelMessage.content = responseText;
  history_.push_back(modelMessage);
  
  return responseText;
}

//------------------------------------------------------------------------------
// sendMessageWithImage - Multimodal image + text
//------------------------------------------------------------------------------
std::string HybridLiteRTLM::sendMessageWithImage(
    const std::string& message,
    const std::string& imagePath) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
#ifdef LITERT_LM_ENABLED
  // Load image using stb_image
  int width, height, channels;
  unsigned char* img = stbi_load(imagePath.c_str(), &width, &height, &channels, 3); // Force 3 channels (RGB)
  if (img == nullptr) {
    throw std::runtime_error("Failed to load image from path: " + imagePath);
  }

  // Create input tensor/buffer for the engine.
  // Note: The exact API for passing image data depends on the LiteRT-LM version.
  // Assuming a structure that accepts raw bytes and dimensions.
  litert::lm::UserMessage lm_message;
  lm_message.role = "user";
  
  // Construct multimodal content
  // Option A: If UserMessage supports a list of content parts
  litert::lm::ContentPart textPart;
  textPart.type = litert::lm::ContentType::TEXT;
  textPart.text = message;
  lm_message.parts.push_back(textPart);

  litert::lm::ContentPart imagePart;
  imagePart.type = litert::lm::ContentType::IMAGE;
  imagePart.image.width = width;
  imagePart.image.height = height;
  imagePart.image.channels = channels;
  imagePart.image.data = std::vector<uint8_t>(img, img + (width * height * channels));
  lm_message.parts.push_back(imagePart);
  
  stbi_image_free(img);

  auto response = conversation_->SendMessage(lm_message);
  if (!response.ok()) {
    throw std::runtime_error("Multimodal inference failed: " + 
        std::string(response.status().message()));
  }
  
  // Add to history (metadata only)
  Message userMessage;
  userMessage.role = Role::USER;
  userMessage.content = message + " [Image]"; 
  history_.push_back(userMessage);
  
  Message modelMessage;
  modelMessage.role = Role::MODEL;
  modelMessage.content = response->content;
  history_.push_back(modelMessage);
  
  return response->content;
  
#else
  // iOS: LiteRT-LM SDK not yet available, throw clear error
  throw std::runtime_error(
      "sendMessageWithImage is not supported on iOS. "
      "LiteRT-LM iOS SDK is not yet available. "
      "Please use text-only sendMessage() for now.");
#endif
}

//------------------------------------------------------------------------------
// sendMessageWithAudio - Multimodal audio + text
//------------------------------------------------------------------------------
std::string HybridLiteRTLM::sendMessageWithAudio(
    const std::string& message,
    const std::string& audioPath) {
  
  std::lock_guard<std::mutex> lock(mutex_);
  ensureLoaded();
  
#ifdef LITERT_LM_ENABLED
  // Load audio file
  std::ifstream audioFile(audioPath, std::ios::binary);
  if (!audioFile) {
      throw std::runtime_error("Failed to open audio file: " + audioPath);
  }
  
  // Simple WAV header skip (simplistic, assuming standard header size for now or raw)
  // Ideally use a WAV parsing library or miniaudio if available.
  // For this implementation, we read the whole file.
  std::vector<uint8_t> audioData((std::istreambuf_iterator<char>(audioFile)), std::istreambuf_iterator<char>());
  
  litert::lm::UserMessage lm_message;
  lm_message.role = "user";
  
  litert::lm::ContentPart textPart;
  textPart.type = litert::lm::ContentType::TEXT;
  textPart.text = message;
  lm_message.parts.push_back(textPart);

  litert::lm::ContentPart audioPart;
  audioPart.type = litert::lm::ContentType::AUDIO;
  audioPart.audio.data = audioData;
  // Metadata like sample rate might be needed:
  // audioPart.audio.sample_rate = 16000; 
  lm_message.parts.push_back(audioPart);

  auto response = conversation_->SendMessage(lm_message);
  if (!response.ok()) {
    throw std::runtime_error("Audio inference failed: " + 
        std::string(response.status().message()));
  }
  
  Message userMessage;
  userMessage.role = Role::USER;
  userMessage.content = message + " [Audio]";
  history_.push_back(userMessage);
  
  Message modelMessage;
  modelMessage.role = Role::MODEL;
  modelMessage.content = response->content;
  history_.push_back(modelMessage);
  
  return response->content;
  
#else
  // iOS: LiteRT-LM SDK not yet available, throw clear error
  throw std::runtime_error(
      "sendMessageWithAudio is not supported on iOS. "
      "LiteRT-LM iOS SDK is not yet available. "
      "Please use text-only sendMessage() for now.");
#endif
}

//------------------------------------------------------------------------------
// sendMessageAsync - Streaming token generation
//------------------------------------------------------------------------------
void HybridLiteRTLM::sendMessageAsync(
    const std::string& message,
    const std::function<void(std::string, bool)>& onToken) {
  
  // Note: We don't hold the lock during the entire async operation
  // to avoid blocking other operations. The callback may be invoked
  // from a different thread depending on LiteRT-LM's implementation.
  
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ensureLoaded();
  }
  
#ifdef LITERT_LM_ENABLED
  // Add user message to history before starting
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Message userMessage;
    userMessage.role = Role::USER;
    userMessage.content = message;
    history_.push_back(userMessage);
  }
  
  litert::lm::UserMessage lm_message;
  lm_message.role = "user";
  lm_message.content = message;
  
  std::string fullResponse;
  
  // The callback needs to be carefully managed for thread safety
  auto status = conversation_->SendMessageAsync(
    lm_message,
    [this, &onToken, &fullResponse](const std::string& token, bool isDone) {
      fullResponse += token;
      
      // Invoke the JS callback (Nitro handles thread marshalling)
      onToken(token, isDone);
      
      if (isDone) {
        // Add complete response to history
        std::lock_guard<std::mutex> lock(mutex_);
        Message modelMessage;
        modelMessage.role = Role::MODEL;
        modelMessage.content = fullResponse;
        history_.push_back(modelMessage);
      }
    }
  );
  
  if (!status.ok()) {
    // Remove user message since inference failed
    std::lock_guard<std::mutex> lock(mutex_);
    if (!history_.empty()) {
      history_.pop_back();
    }
    throw std::runtime_error("Async inference failed: " + 
        std::string(status.message()));
  }
  
#else
  // Stub: Simulate streaming by calling sendMessage and splitting response
  std::string fullResponse;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Add user message
    Message userMessage;
    userMessage.role = Role::USER;
    userMessage.content = message;
    history_.push_back(userMessage);
    
    fullResponse = "[LiteRT-LM Stub] Streaming response placeholder. You said: " + message;
  }
  
  // Simulate token-by-token streaming
  std::string currentWord;
  for (size_t i = 0; i < fullResponse.length(); i++) {
    char c = fullResponse[i];
    currentWord += c;
    
    if (c == ' ' || c == '\n' || i == fullResponse.length() - 1) {
      bool isDone = (i == fullResponse.length() - 1);
      onToken(currentWord, isDone);
      currentWord.clear();
    }
  }
  
  // Add model response to history
  {
    std::lock_guard<std::mutex> lock(mutex_);
    Message modelMessage;
    modelMessage.role = Role::MODEL;
    modelMessage.content = fullResponse;
    history_.push_back(modelMessage);
  }
#endif
}

//------------------------------------------------------------------------------
// getHistory - Return conversation history
//------------------------------------------------------------------------------
std::vector<Message> HybridLiteRTLM::getHistory() {
  std::lock_guard<std::mutex> lock(mutex_);
  return history_;
}

//------------------------------------------------------------------------------
// resetConversation - Clear KV cache, keep engine
//------------------------------------------------------------------------------
void HybridLiteRTLM::resetConversation() {
  std::lock_guard<std::mutex> lock(mutex_);
  
#ifdef LITERT_LM_ENABLED
  // Destroy old conversation and create a new one
  // This clears the KV cache but keeps the (expensive) Engine loaded
  if (engine_) {
    conversation_.reset();
    createNewConversation();
  }
#endif
  
  history_.clear();
}

//------------------------------------------------------------------------------
// isReady - Check if model is loaded
//------------------------------------------------------------------------------
bool HybridLiteRTLM::isReady() {
  std::lock_guard<std::mutex> lock(mutex_);
  return isLoaded_;
}

//------------------------------------------------------------------------------
// getStats - Return last generation statistics
//------------------------------------------------------------------------------
GenerationStats HybridLiteRTLM::getStats() {
  std::lock_guard<std::mutex> lock(mutex_);
  return lastStats_;
}

//------------------------------------------------------------------------------
// close - Release all native resources
//------------------------------------------------------------------------------
void HybridLiteRTLM::close() {
  std::lock_guard<std::mutex> lock(mutex_);
  
#ifdef LITERT_LM_ENABLED
  // Release in reverse order of creation
  conversation_.reset();
  engine_.reset();
#endif
  
  isLoaded_ = false;
  history_.clear();
}

} // namespace margelo::nitro::litertlm
