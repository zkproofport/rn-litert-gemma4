package com.margelo.nitro.dev.litert.litertlm

import java.util.Collections
import java.util.WeakHashMap
import android.util.Log

/**
 * Global registry to track active LiteRTLM instances.
 * Used for memory trimming and cleanup.
 */
object LiteRTLMRegistry {
    private const val TAG = "LiteRTLMRegistry"
    
    // Use WeakSet-like structure to prevent leaks
    private val instances = Collections.newSetFromMap(WeakHashMap<HybridLiteRTLM, Boolean>())

    fun register(instance: HybridLiteRTLM) {
        synchronized(instances) {
            instances.add(instance)
        }
    }

    fun onTrimMemory(level: Int) {
        Log.w(TAG, "Received memory warning (level=$level). Releasing resources...")
        synchronized(instances) {
            instances.forEach { it.close() }
            // Note: We don't clear the set here, as close() should be idempotent
            // and the instance might still be ref-counted by JS. 
            // We just ensure the HEAVY native resources are gone.
        }
    }
}
