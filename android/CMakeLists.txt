cmake_minimum_required(VERSION 3.18.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define the library name - must match what Nitrogen expects
project(LiteRTLM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the shared library (main entry point)
add_library(
    LiteRTLM
    SHARED
    ../cpp/cpp-adapter.cpp
    # Additional sources are added by autolinking.cmake below
)

# Allow undefined symbols - they will be resolved at runtime when the app
# loads the NitroModules shared library. This is required because we're
# building a library that depends on NitroModules symbols which are only
# available at runtime.
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-shlib-undefined,-z,max-page-size=16384")

# Include Nitrogen autolinking - this adds all generated sources and links
include(${CMAKE_SOURCE_DIR}/../nitrogen/generated/android/LiteRTLM+autolinking.cmake)

# Android system libraries
target_link_libraries(
    LiteRTLM
    android
    log
)
