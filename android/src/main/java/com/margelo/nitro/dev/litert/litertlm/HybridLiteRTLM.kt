///
/// HybridLiteRTLM.kt
/// Kotlin implementation of LiteRTLM HybridObject using LiteRT-LM Android SDK.
///

package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import dev.litert.litertlm.LiteRTLMInitProvider
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ConversationConfig
import com.margelo.nitro.dev.litert.litertlm.Backend
import com.margelo.nitro.dev.litert.litertlm.GenerationStats
import com.margelo.nitro.dev.litert.litertlm.HybridLiteRTLMSpec
import com.margelo.nitro.dev.litert.litertlm.LLMConfig
import com.margelo.nitro.dev.litert.litertlm.Message
import com.margelo.nitro.dev.litert.litertlm.Role
import com.margelo.nitro.core.Promise

// Alias to avoid confusion with our generated Message type
typealias LiteRTMessage = com.google.ai.edge.litertlm.Message

/**
 * Kotlin implementation of LiteRTLM using the LiteRT-LM Android SDK.
 * This class bridges between React Native (via Nitro) and the Google LiteRT-LM Engine.
 */
@DoNotStrip
@Keep
class HybridLiteRTLM : HybridLiteRTLMSpec() {

    companion object {
        private const val TAG = "HybridLiteRTLM"
    }

    init {
        LiteRTLMRegistry.register(this)
    }

    // LiteRT-LM Engine and Conversation
    private var engine: Engine? = null
    private var conversation: Conversation? = null

    // Conversation history for getHistory()
    private val history = mutableListOf<Message>()

    // Last generation stats
    private var lastStats = GenerationStats(
        promptTokens = 0.0,
        completionTokens = 0.0,
        totalTokens = 0.0,
        timeToFirstToken = 0.0,
        totalTime = 0.0,
        tokensPerSecond = 0.0
    )

    // Configuration
    private var backend: Backend = Backend.GPU
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxTokens: Int = 1024

    override val memorySize: Long
        get() = 1024L * 1024L * 1024L // ~1GB (models are large)

    // -------------------------------------------------------------------------
    // loadModel - Initialize LiteRT-LM Engine and Conversation
    // -------------------------------------------------------------------------
    override fun loadModel(modelPath: String, config: LLMConfig?): Promise<Unit> {
        return Promise.parallel {
            Log.i(TAG, "loadModel: $modelPath")

            // Clean up existing resources
            close()

            // Apply configuration
            config?.let { cfg ->
                cfg.backend?.let { backend = it }
                cfg.temperature?.let { temperature = it }
                cfg.topK?.let { topK = it.toInt() }
                cfg.topP?.let { topP = it }
                cfg.maxTokens?.let { maxTokens = it.toInt() }
            }

            try {
                // Map our Backend enum to LiteRT-LM Backend enum
                val lmBackend = when (backend) {
                    Backend.GPU -> com.google.ai.edge.litertlm.Backend.GPU
                    Backend.NPU -> {
                        Log.i(TAG, "NPU backend requested - requires hardware support")
                        com.google.ai.edge.litertlm.Backend.NPU
                    }
                    else -> com.google.ai.edge.litertlm.Backend.CPU
                }
                
                // Vision backend: hardcoded to GPU (required by Gemma 3n)
                val lmVisionBackend = com.google.ai.edge.litertlm.Backend.GPU
                    
                // Audio backend: hardcoded to CPU (optimal for audio processing)
                val lmAudioBackend = com.google.ai.edge.litertlm.Backend.CPU

                Log.i(TAG, "Backend config: main=$lmBackend, vision=$lmVisionBackend (hardcoded), audio=$lmAudioBackend (hardcoded)")

                // Get cache directory from application context
                val cacheDirectory = LiteRTLMInitProvider.applicationContext?.cacheDir?.absolutePath
                Log.i(TAG, "Using cache directory: $cacheDirectory")

                // Create Engine configuration
                val engineConfig = EngineConfig(
                    modelPath = modelPath,
                    backend = lmBackend,
                    visionBackend = lmVisionBackend,
                    audioBackend = lmAudioBackend,
                    maxNumTokens = maxTokens,
                    cacheDir = cacheDirectory
                )

                // Initialize Engine
                engine = Engine(engineConfig).also { it.initialize() }
                Log.i(TAG, "Engine created and initialized successfully")

                // Create Conversation
                createNewConversation()
                Log.i(TAG, "Conversation created successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: ${e.message}", e)
                throw RuntimeException("Failed to load model: ${e.message}", e)
            }
        }
    }

    // -------------------------------------------------------------------------
    // sendMessage - Helper for one-shot generation (internally uses Async)
    // -------------------------------------------------------------------------
    override fun sendMessage(message: String): Promise<String> {
        // Implement Promise-based sendMessage using suspend coroutine logic wrapped in Promise
        // Since Promise.parallel expects a blocking block returning T, 
        // and sendMessageAsync is callback-based, we need to bridge them.
        // HOWEVER, we can just use the synchronous `sendMessage` API of the SDK 
        // inside the `Promise.parallel` block, which moves it off the main thread!
        return Promise.parallel {
            ensureLoaded()

            // Add user message to history
            history.add(Message(Role.USER, message))
            Log.i(TAG, "sendMessage (Promise): $message")
            
            // Blocking inference (safe here because we are in Promise.parallel worker thread)
            val userMsg = LiteRTMessage.of(message)
            val responseMsg = conversation!!.sendMessage(userMsg)
            
            // Extract text
            val response = responseMsg.contents
                .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
                .joinToString("") { it.text } 
            
            // Add model response to history
            history.add(Message(Role.MODEL, response))
            
            // Update stats
            lastStats = GenerationStats(
                promptTokens = message.length / 4.0,
                completionTokens = response.length / 4.0,
                totalTokens = (message.length + response.length) / 4.0,
                timeToFirstToken = 0.0,
                totalTime = 0.0,
                tokensPerSecond = 0.0
            )
            
            response // Return the string
        }
    }

    // -------------------------------------------------------------------------
    // sendMessageAsync - Streaming inference
    // -------------------------------------------------------------------------
    override fun sendMessageAsync(message: String, onToken: (String, Boolean) -> Unit) {
        // This is already async (void return), so we execute immediately on the calling thread
        // (which is the Nitro specialized thread, not Main).
        // The SDK's sendMessageAsync is non-blocking anyway.
        ensureLoaded()

        // Add user message to history
        history.add(Message(Role.USER, message))
        Log.d(TAG, "sendMessageAsync: $message")

        val fullResponseBuilder = StringBuilder()
        
        // Define callback
        val listener = object : com.google.ai.edge.litertlm.MessageCallback {
             override fun onMessage(responseMsg: LiteRTMessage) {
                val chunk = responseMsg.contents
                    .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
                    .joinToString("") { it.text }

                onToken(chunk, false)
                
                if (chunk.isNotEmpty()) {
                    fullResponseBuilder.append(chunk)
                }
            }
            
            override fun onDone() {
                onToken("", true)
                val fullResponse = fullResponseBuilder.toString()
                history.add(Message(Role.MODEL, fullResponse))
                Log.d(TAG, "sendMessageAsync done. Length: ${fullResponse.length}")
            }
            
            override fun onError(t: Throwable) {
                Log.e(TAG, "Async generation failed", t)
                onToken("Error: ${t.message}", true)
            }
        }

        try {
            val userMsg = LiteRTMessage.of(message)
            conversation!!.sendMessageAsync(userMsg, listener)
        } catch (e: Exception) {
            Log.e(TAG, "Failed into initiate async generation", e)
            onToken("Error: ${e.message}", true)
        }
    }

    // -------------------------------------------------------------------------
    // Multimodal methods
    // -------------------------------------------------------------------------
    override fun sendMessageWithImage(message: String, imagePath: String): Promise<String> {
        return Promise.parallel {
             // TODO: Implement image loading from path
            throw RuntimeException("Multimodal (Image) not yet implemented in this wrapper")
        }
    }

    override fun sendMessageWithAudio(message: String, audioPath: String): Promise<String> {
        return Promise.parallel {
            // TODO: Implement audio loading from path
            throw RuntimeException("Multimodal (Audio) not yet implemented in this wrapper")
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    override fun getHistory(): Array<Message> {
        return history.toTypedArray()
    }

    override fun resetConversation() {
        history.clear()
        createNewConversation()
    }

    override fun isReady(): Boolean {
        return isLoaded_
    }
    
    // Property backing field for isReady check
    private val isLoaded_: Boolean
        get() = engine != null

    override fun getStats(): GenerationStats {
        return lastStats
    }

    override fun close() {
        Log.d(TAG, "Closing resources")
        try {
            conversation = null
            engine = null // Engine destructor should handle cleanup
            // In C++ we'd close explicitly, Kotlin GC helps but explicit close method is better if SDK has it
        } catch (e: Exception) {
            Log.e(TAG, "Error closing resources", e)
        }
    }

    private fun ensureLoaded() {
        if (engine == null) {
            throw RuntimeException("LiteRTLM: No model loaded. Call loadModel() first.")
        }
    }

    private fun createNewConversation() {
        ensureLoaded()
        // Dispose old conversation if needed
        conversation = engine!!.createConversation()
    }
}
