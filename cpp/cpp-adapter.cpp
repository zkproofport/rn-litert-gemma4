///
/// cpp-adapter.cpp
/// JNI Entry Point - Required by Nitrogen to register Kotlin HybridObjects
///
/// Updated for react-native-nitro-modules v0.34+:
/// Uses facebook::jni::initialize() directly with registerAllNatives().
///

#include <jni.h>
#include <fbjni/fbjni.h>
#include "LiteRTLMOnLoad.hpp"

// JNI_OnLoad is called when the native library is loaded via System.loadLibrary()
// This is where we initialize the Nitrogen bridge and register all Kotlin HybridObjects.
// The new v0.34 API allows registering custom C++ native JNI classes/functions
// alongside Nitrogen's auto-generated registrations.
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    return facebook::jni::initialize(vm, []() {
        margelo::nitro::litertlm::registerAllNatives();
    });
}
