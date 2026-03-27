///
/// HybridLiteRTLM.kt
/// Kotlin implementation of LiteRTLM HybridObject using LiteRT-LM Android SDK.
///

package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import android.os.Debug
import android.app.ActivityManager
import android.content.Context
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
import com.google.ai.edge.litertlm.Content


// Alias to avoid confusion with our generated Message type
// Alias to avoid confusion
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
        private val initLock = Any()
        
        /**
         * Initialize the native library.
         * Must be called from Application.onCreate() to register the HybridObject.
         */
        fun initialize() {
            try {
                // Call generated internal OnLoad to load the library
                LiteRTLMOnLoad.initializeNative()
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to initialize LiteRTLM native library", e)
            }
        }
    }

    init {
        LiteRTLMRegistry.register(this)
    }

    // LiteRT-LM Engine and Conversation
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    
    @Volatile
    private var isClosed = false

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
    private var systemPrompt: String? = null

    override val memorySize: Long
        get() = 1024L * 1024L * 1024L // ~1GB (models are large)

    // -------------------------------------------------------------------------
    // loadModel - Initialize LiteRT-LM Engine and Conversation
    // -------------------------------------------------------------------------
    override fun loadModel(modelPath: String, config: LLMConfig?): Promise<Unit> {
        return Promise.parallel {
            // Serialize initialization to prevent OOM from concurrent loads
            synchronized(initLock) {
                if (isClosed) {
                    throw RuntimeException("Cannot load model: LiteRTLM instance is closed")
                }
                
                Log.i(TAG, "loadModel: $modelPath")
    
                // Clean up existing resources
                // We call internal cleanup that doesn't set isClosed
                cleanupInternal()
    
                // Apply configuration
                config?.let { cfg ->
                    cfg.backend?.let { backend = it }
                    cfg.temperature?.let { temperature = it }
                    cfg.topK?.let { topK = it.toInt() }
                    cfg.topP?.let { topP = it }
                    cfg.maxTokens?.let { maxTokens = it.toInt() }
                    cfg.systemPrompt?.let { systemPrompt = it }
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
    
                    if (isClosed) return@synchronized
                    
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
            Log.e(TAG, "Failed to initiate async generation", e)
            onToken("Error: ${e.message}", true)
        }
    }

    // -------------------------------------------------------------------------
    // Multimodal methods
    // -------------------------------------------------------------------------
    
    /**
     * Resize image if dimensions exceed maxDimension to prevent OOM.
     * Gemma 3n's vision encoder is optimized for 512x512 or 1024x1024.
     * Passing larger images can spike memory 500MB+.
     */
    private fun resizeImageIfNeeded(imagePath: String, maxDimension: Int = 1024): String {
        val originalBitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
            ?: throw RuntimeException("Failed to decode image: $imagePath")

        val width = originalBitmap.width
        val height = originalBitmap.height

        // If already within bounds, return original path
        if (width <= maxDimension && height <= maxDimension) {
            originalBitmap.recycle()
            return imagePath
        }

        Log.i(TAG, "Resizing image from ${width}x${height} to fit ${maxDimension}px")

        val scale = maxDimension.toFloat() / maxOf(width, height)
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        val resizedBitmap = android.graphics.Bitmap.createScaledBitmap(originalBitmap, newWidth, newHeight, true)
        originalBitmap.recycle()

        // Save to temp file
        val cacheDir = LiteRTLMInitProvider.applicationContext?.cacheDir
            ?: throw RuntimeException("Application context not available for image resizing")
        val tempFile = java.io.File(cacheDir, "resized_${System.currentTimeMillis()}.jpg")
        java.io.FileOutputStream(tempFile).use { out ->
            resizedBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
        }
        resizedBitmap.recycle()

        Log.i(TAG, "Resized image saved to: ${tempFile.absolutePath} (${newWidth}x${newHeight})")
        return tempFile.absolutePath
    }

    override fun sendMessageWithImage(message: String, imagePath: String): Promise<String> {
        return Promise.parallel {
            ensureLoaded()
            Log.i(TAG, "sendMessageWithImage: $message, path=$imagePath")

            // Resize image to prevent OOM on high-resolution photos
            val processedImagePath = resizeImageIfNeeded(imagePath)

            // Create multimodal message
            // Use factory method Message.of passing a list of Content
            val textContent = Content.Text(message)
            
            val contentList = listOf(
                textContent,
                Content.ImageFile(processedImagePath)
            )

            val userMsg = LiteRTMessage.of(contentList)

            // Add to history
            history.add(Message(Role.USER, "$message [Image]"))

            val responseMsg = conversation!!.sendMessage(userMsg)
            
            val response = responseMsg.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }

            history.add(Message(Role.MODEL, response))
            
            response
        }
    }

    override fun downloadModel(url: String, fileName: String, onProgress: ((Double) -> Unit)?): Promise<String> {
        return Promise.parallel {
            Log.i(TAG, "downloadModel: $url -> $fileName")
            
            val context = LiteRTLMInitProvider.applicationContext ?: throw RuntimeException("Context not available")
            val modelsDir = java.io.File(context.filesDir, "models")
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }
            
            val modelFile = java.io.File(modelsDir, fileName)
            val tempFile = java.io.File(modelsDir, "$fileName.tmp")
            
            // Check if file exists and has content
            if (modelFile.exists() && modelFile.length() > 0) {
                Log.i(TAG, "Model already exists at: ${modelFile.absolutePath}")
                onProgress?.invoke(1.0)
                return@parallel modelFile.absolutePath
            }
            
            Log.i(TAG, "Downloading model to temp file: ${tempFile.absolutePath}")
            onProgress?.invoke(0.0)
            
            try {
                val connection = java.net.URL(url).openConnection() as java.net.HttpURLConnection
                connection.connectTimeout = 15000 // 15s
                connection.readTimeout = 0 // Infinite for large files
                connection.doInput = true
                connection.connect()
                
                if (connection.responseCode != java.net.HttpURLConnection.HTTP_OK) {
                    throw RuntimeException("Failed to download model: HTTP ${connection.responseCode}")
                }
                
                val contentLength = connection.contentLengthLong // Use long for large files
                val input = connection.inputStream
                val output = java.io.FileOutputStream(tempFile)
                
                val buffer = ByteArray(8 * 1024)
                var bytesRead: Int
                var totalBytesRead = 0L
                var lastProgressUpdate = 0L
                
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead
                    
                    if (contentLength > 0 && onProgress != null) {
                        val currentTime = System.currentTimeMillis()
                        // Update roughly every 100ms to avoid flooding JS bridge
                        if (currentTime - lastProgressUpdate > 100) {
                            val progress = totalBytesRead.toDouble() / contentLength.toDouble()
                            onProgress(progress)
                            lastProgressUpdate = currentTime
                        }
                    }
                }
                
                output.flush()
                output.close()
                input.close()
                connection.disconnect()
                
                // Atomic rename
                if (tempFile.renameTo(modelFile)) {
                    Log.i(TAG, "Download complete and renamed to: ${modelFile.absolutePath}")
                    onProgress?.invoke(1.0)
                    return@parallel modelFile.absolutePath
                } else {
                    throw RuntimeException("Failed to rename temp file to model file")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                if (tempFile.exists()) {
                    tempFile.delete()
                }
                throw RuntimeException("Download failed: ${e.message}", e)
            }
        }
    }

    override fun deleteModel(fileName: String): Promise<Unit> {
        return Promise.parallel {
            Log.i(TAG, "deleteModel: $fileName")
            val context = LiteRTLMInitProvider.applicationContext ?: throw RuntimeException("Context not available")
            val modelsDir = java.io.File(context.filesDir, "models")
            val modelFile = java.io.File(modelsDir, fileName)
            
            if (modelFile.exists()) {
                val deleted = modelFile.delete()
                if (deleted) {
                    Log.i(TAG, "Deleted model: ${modelFile.absolutePath}")
                    // Ensure engine references are cleared if they point to this file
                    // We use cleanupInternal() which releases resources WITHOUT marking the instance as closed.
                    if (engine != null) {
                        Log.i(TAG, "Cleaning up engine after deleting model file.")
                        cleanupInternal()
                    }
                } else {
                    Log.e(TAG, "Failed to delete model: ${modelFile.absolutePath}")
                    throw RuntimeException("Failed to delete model: ${modelFile.absolutePath}")
                }
            } else {
                Log.w(TAG, "Model not found for deletion: ${modelFile.absolutePath}")
            }
        }
    }

    override fun sendMessageWithAudio(message: String, audioPath: String): Promise<String> {
        return Promise.parallel {
            ensureLoaded()
            Log.i(TAG, "sendMessageWithAudio: $message, path=$audioPath")

            // Load audio
            
            val contentList = listOf(
                Content.Text(message),
                Content.AudioFile(audioPath)
            )
            
            val userMsg = LiteRTMessage.of(contentList)

            history.add(Message(Role.USER, "$message [Audio]"))
            
            val responseMsg = conversation!!.sendMessage(userMsg)
            
            val response = responseMsg.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }
                
            history.add(Message(Role.MODEL, response))
            
            response
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

    override fun getMemoryUsage(): MemoryUsage {
        // Native heap: allocated bytes from Debug APIs (most accurate for native allocations)
        val nativeHeapBytes = Debug.getNativeHeapAllocatedSize().toDouble()

        // Process RSS: read from /proc/self/status (VmRSS) in kB
        var residentBytes = 0.0
        try {
            java.io.File("/proc/self/status").forEachLine { line ->
                if (line.startsWith("VmRSS:")) {
                    val kb = line.substringAfter("VmRSS:").trim().split("\\s+".toRegex())[0].toDoubleOrNull()
                    if (kb != null) {
                        residentBytes = kb * 1024.0
                    }
                    return@forEachLine
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read /proc/self/status: ${e.message}")
        }

        // Available memory and low-memory flag from ActivityManager
        var availableMemoryBytes = 0.0
        var isLowMemory = false
        try {
            val context = LiteRTLMInitProvider.applicationContext
            if (context != null) {
                val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                val memInfo = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memInfo)
                availableMemoryBytes = memInfo.availMem.toDouble()
                isLowMemory = memInfo.lowMemory
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to get ActivityManager memory info: ${e.message}")
        }

        return MemoryUsage(
            nativeHeapBytes = nativeHeapBytes,
            residentBytes = residentBytes,
            availableMemoryBytes = availableMemoryBytes,
            isLowMemory = isLowMemory
        )
    }

    override fun close() {
        Log.d(TAG, "Closing resources")
        isClosed = true
        cleanupInternal()
    }

    private fun cleanupInternal() {
        try {
            conversation = null
            // Explicitly close engine if it supports it to free native memory immediately
            // Assuming Engine implements AutoCloseable or has close()
            if (engine is AutoCloseable) {
                (engine as AutoCloseable).close()
            } else {
                 // Try reflection or just null it if no close method
                try {
                    engine?.javaClass?.getMethod("close")?.invoke(engine)
                } catch (e: Exception) {
                    // Method not found, rely on GC
                }
            }
            engine = null 
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
        // Apply system prompt/instruction if set
        systemPrompt?.let { prompt ->
            if (prompt.isNotEmpty()) {
                try {
                    // Send system instruction as the first turn to prime the conversation.
                    // LiteRT-LM's Conversation API handles chat template formatting,
                    // including Gemma's <start_of_turn>system block.
                    val systemMsg = LiteRTMessage.of(listOf(Content.Text(prompt)))
                    conversation!!.sendMessage(systemMsg)
                    Log.i(TAG, "System prompt applied (${prompt.length} chars)")
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to apply system prompt: ${e.message}")
                }
            }
        }
    }


}
