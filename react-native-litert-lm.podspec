require "json"

package = JSON.parse(File.read(File.join(__dir__, "package.json")))

Pod::Spec.new do |s|
  s.name         = "react-native-litert-lm"
  s.version      = package["version"]
  s.summary      = package["description"]
  s.homepage     = package["homepage"]
  s.license      = package["license"]
  s.authors      = package["author"]
  s.platforms    = { :ios => "15.0" }
  s.source       = { :git => package["repository"]["url"], :tag => "#{s.version}" }
  
  s.swift_version = '5.0'

  s.source_files = [
    # Implementation (C++)
    "cpp/**/*.{hpp,cpp,h,c}",
    # Autolinking (Objective-C++)
    "ios/**/*.{m,mm}",
    # Nitrogen generated iOS bridge
    "nitrogen/generated/ios/**/*.{mm,swift}",
  ]

  # Exclude Android-only JNI files + vendored upstream headers from iOS build
  s.exclude_files = [
    "cpp/cpp-adapter.cpp",
    "cpp/upstream/**/*",
  ]

  # Prebuilt LiteRT-LM C engine (static library built from Bazel //c:engine target).
  # Downloaded from GitHub releases by postinstall.js, or built locally via:
  #   scripts/build-ios-engine.sh
  #
  # GemmaModelConstraintProvider: dynamic library shipped by Google in the
  # LiteRT-LM v0.10.2 source tree (`prebuilt/{ios_arm64,ios_sim_arm64}/`).
  # The upstream wrapper's static framework stubs the FST constraint provider
  # ("Gemma Constraint Provider is STUBBED/DISABLED") which makes
  # `enable_constrained_decoding=true` fail. We removed the stub object from
  # the static archive and link this dylib instead, restoring real FST-based
  # constrained decoding for Gemma 4 native function calling.
  # GPU acceleration: LiteRT-LM dlopen()s these dylibs at runtime to enable
  # Metal/GPU. Without them: stderr shows "GPU accelerator could not be loaded"
  # and inference falls back to CPU (~1.4 token/sec instead of 30+).
  s.vendored_frameworks = [
    'ios/Frameworks/LiteRTLM.xcframework',
    # GemmaModelConstraintProvider exposes C API symbols needed at link time
    # (referenced by data processors). Keep as framework wrap so Pods links it.
    # The other GPU dylibs (LiteRt, MetalAccelerator, TopKMetalSampler) are
    # dlopen'd by name at runtime; they live as plain .dylib in ios/dylibs/
    # and customer must copy to App.app/Frameworks/.
    'ios/Frameworks/GemmaModelConstraintProvider.xcframework',
  ]

  # GPU acceleration plain dylibs (in ios/dylibs/{ios_arm64,ios_sim_arm64}/).
  # LiteRT-LM's gpu_registry calls dlopen("libLiteRtMetalAccelerator.dylib")
  # with the plain library name, so the dylibs must live under
  # @rpath/<plain-name>.dylib (i.e. App.app/Frameworks/<plain>.dylib), NOT
  # inside *.framework bundles. The customer app must add a Run Script build
  # phase (or scripts/copy-litert-gpu-dylibs.sh) to copy the right slice into
  # App.app/Frameworks/ and re-codesign. Without this, sendMessage runs ~25x
  # slower on CPU instead of using Metal.
  s.preserve_paths = ['ios/dylibs/**/*.dylib']

  # Common include paths shared between sim + device.
  # cpp/upstream/common provides:
  #   - runtime/  -> upstream LiteRT-LM runtime/ source tree (symlink to /tmp/LiteRT-LM-src/runtime)
  #   - c/        -> upstream c/ (C API source)
  #   - schema/   -> upstream schema/ (model file format)
  #   - absl/     -> Abseil headers from bazel external/com_google_absl/absl
  #   - nlohmann/ -> nlohmann::json single-header from bazel external/nlohmann_json
  # cpp/upstream/include-sim and include-dev provide arch-specific generated
  # proto headers (`runtime/proto/*.pb.h`) from bazel-out.
  common_includes = [
    '"$(PODS_TARGET_SRCROOT)/cpp"',
    '"$(PODS_TARGET_SRCROOT)/cpp/include"',
    '"$(PODS_TARGET_SRCROOT)/cpp/upstream/common"',
    # protobuf-src/google/protobuf/... — protobuf source uses `google/protobuf/...`
    # so the inner `src/` directory must itself be on the include path.
    '"$(PODS_TARGET_SRCROOT)/cpp/upstream/common/protobuf-src"',
    '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/shared/c++"',
    '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/ios"',
  ]

  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
    'CLANG_CXX_LIBRARY' => 'libc++',
    # Default search paths (must include $(inherited) so NitroModules + other
    # Pod-public headers stay resolvable).
    'HEADER_SEARCH_PATHS' => (['$(inherited)'] + common_includes).join(' '),
    # iOS simulator: pull generated protos from ios_sim_arm64 bazel-out.
    'HEADER_SEARCH_PATHS[sdk=iphonesimulator*]' =>
      (['$(inherited)', '"$(PODS_TARGET_SRCROOT)/cpp/upstream/include-sim"'] + common_includes).join(' '),
    # iOS device: pull generated protos from ios_arm64 bazel-out.
    'HEADER_SEARCH_PATHS[sdk=iphoneos*]' =>
      (['$(inherited)', '"$(PODS_TARGET_SRCROOT)/cpp/upstream/include-dev"'] + common_includes).join(' '),
    # NOTE: -force_load on LiteRTLM is set in the consumer Podfile's
    # post_install (BUILT_PRODUCTS_DIR resolves correctly there). Setting it
    # here in pod_target_xcconfig points at the wrong build dir.
    'OTHER_LDFLAGS' => '$(inherited) -ObjC',
  }

  # Load nitrogen autolinking
  load 'nitrogen/generated/ios/LiteRTLM+autolinking.rb'
  add_nitrogen_files(s)

  # Core React Native dependencies
  s.dependency 'React-jsi'
  s.dependency 'React-callinvoker'
  s.dependency 'ReactCommon/turbomodule/core'

  # Apple frameworks needed by LiteRT-LM engine.
  # Metal/MPS: GPU inference, Accelerate: BLAS/LAPACK, CoreML: delegate,
  # AVFoundation: required by miniaudio (audio session routing/interruption
  # handlers compiled into the LiteRTLM XCFramework). Without this, apps
  # that don't already pull AVFoundation in (e.g. via Firebase) fail at link
  # time with `Undefined symbols: _AVAudioSession*`.
  s.frameworks = ['Metal', 'MetalPerformanceShaders', 'Accelerate', 'CoreML', 'CoreGraphics', 'AVFoundation']
  s.libraries = ['c++']

  install_modules_dependencies(s)
end

