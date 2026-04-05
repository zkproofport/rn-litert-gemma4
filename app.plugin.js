/**
 * Expo config plugin for react-native-litert-lm.
 *
 * Ensures correct build settings for the LiteRT-LM native module:
 * - Android: minSdkVersion 26, Kotlin 2.3.0 (required by litertlm-android AAR)
 */
const {
  withGradleProperties,
  withProjectBuildGradle,
} = require('@expo/config-plugins');

function withLiteRTLM(config) {
  // Android: Ensure minSdkVersion is at least 26
  config = withGradleProperties(config, (config) => {
    const props = config.modResults;

    // Set minSdkVersion if not already high enough
    const minSdkProp = props.find((p) => p.key === 'android.minSdkVersion');
    if (!minSdkProp) {
      props.push({
        type: 'property',
        key: 'android.minSdkVersion',
        value: '26',
      });
    } else if (parseInt(minSdkProp.value, 10) < 26) {
      minSdkProp.value = '26';
    }

    return config;
  });

  // Android: Pin Kotlin Gradle plugin to 2.3.0
  // The litertlm-android AAR uses Kotlin 2.3.0 metadata (version defined in
  // package.json → litertLm.androidMavenVersion).
  // React Native's default Kotlin version (2.1.0) cannot read this metadata,
  // so we must force the Kotlin Gradle plugin to 2.3.0 in the project-level
  // build.gradle. This ensures the fix survives `expo prebuild --clean`.
  config = withProjectBuildGradle(config, (config) => {
    if (config.modResults.language === 'groovy') {
      const contents = config.modResults.contents;

      // Only add if not already pinned
      if (!contents.includes("kotlin-gradle-plugin:2.3.0")) {
        // Replace the unversioned kotlin-gradle-plugin classpath with a pinned one
        config.modResults.contents = contents.replace(
          "classpath('org.jetbrains.kotlin:kotlin-gradle-plugin')",
          "classpath('org.jetbrains.kotlin:kotlin-gradle-plugin:2.3.0')"
        );
      }
    }

    return config;
  });

  return config;
}

module.exports = withLiteRTLM;
