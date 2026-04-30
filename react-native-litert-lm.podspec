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
    "cpp/**/*.{hpp,cpp,h}",
    # Autolinking (Objective-C++)
    "ios/**/*.{m,mm}",
    # Nitrogen generated iOS bridge
    "nitrogen/generated/ios/**/*.{mm,swift}",
  ]

  # Exclude Android-only JNI files from iOS build
  s.exclude_files = [
    "cpp/cpp-adapter.cpp",
  ]

  # Prebuilt LiteRT-LM C engine (static library built from Bazel //c:engine target).
  # Downloaded from GitHub releases by postinstall.js, or built locally via:
  #   scripts/build-ios-engine.sh
  s.vendored_frameworks = 'ios/Frameworks/LiteRTLM.xcframework'

  s.pod_target_xcconfig = {
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++20',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'HEADER_SEARCH_PATHS' => [
      '"$(PODS_TARGET_SRCROOT)/cpp"',
      '"$(PODS_TARGET_SRCROOT)/cpp/include"',
      '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/shared/c++"',
      '"$(PODS_TARGET_SRCROOT)/nitrogen/generated/ios"',
    ].join(' '),
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

