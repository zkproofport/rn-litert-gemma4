#!/usr/bin/env node
/**
 * postinstall.js
 *
 * Downloads prebuilt LiteRT-LM iOS frameworks from this package's GitHub
 * releases when consumers run `npm install react-native-litert-lm`.
 *
 * Skips download if:
 *   - Not on macOS (iOS builds require macOS)
 *   - Frameworks already exist
 *   - CI environment with SKIP_IOS_FRAMEWORK_DOWNLOAD=1
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');

const PACKAGE_JSON = require('../package.json');
// We pull the prebuilt iOS XCFramework from the upstream
// `react-native-litert-lm` releases. Our package.json version is the fork's
// own version and bears no relation to the upstream release tag, so the
// upstream release tag is configured separately under `upstreamWrapper`.
const UPSTREAM_RELEASE_TAG =
  (PACKAGE_JSON.upstreamWrapper && PACKAGE_JSON.upstreamWrapper.releaseTag) ||
  'v0.3.6';
const GITHUB_REPO = 'hung-yueh/react-native-litert-lm';
const ASSET_NAME = 'LiteRTLM-ios-frameworks.zip';

const SCRIPT_DIR = __dirname;
const PACKAGE_ROOT = path.resolve(SCRIPT_DIR, '..');
const FRAMEWORKS_DIR = path.join(PACKAGE_ROOT, 'ios', 'Frameworks');

function log(msg) {
  console.log(`[react-native-litert-lm] ${msg}`);
}

function shouldSkip() {
  // Skip if not macOS
  if (process.platform !== 'darwin') {
    log('Skipping iOS framework download (not macOS).');
    return true;
  }

  // Skip if explicitly disabled
  if (process.env.SKIP_IOS_FRAMEWORK_DOWNLOAD === '1') {
    log('Skipping iOS framework download (SKIP_IOS_FRAMEWORK_DOWNLOAD=1).');
    return true;
  }

  // Skip if frameworks already exist
  if (fs.existsSync(FRAMEWORKS_DIR) && fs.readdirSync(FRAMEWORKS_DIR).length > 0) {
    log('iOS frameworks already present, skipping download.');
    return true;
  }

  return false;
}

function downloadFile(url, destPath, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) {
      return reject(new Error('Too many redirects'));
    }

    const protocol = url.startsWith('https') ? https : require('http');

    protocol.get(url, { headers: { 'User-Agent': 'react-native-litert-lm' } }, (res) => {
      // Follow redirects
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return downloadFile(res.headers.location, destPath, maxRedirects - 1)
          .then(resolve)
          .catch(reject);
      }

      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} downloading ${url}`));
      }

      const file = fs.createWriteStream(destPath);
      res.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
      file.on('error', reject);
    }).on('error', reject);
  });
}

async function main() {
  if (shouldSkip()) return;

  const releaseUrl = `https://github.com/${GITHUB_REPO}/releases/download/${UPSTREAM_RELEASE_TAG}/${ASSET_NAME}`;

  log(`Downloading iOS frameworks from: ${releaseUrl}`);

  const tmpZip = path.join(PACKAGE_ROOT, '.ios-frameworks-tmp.zip');

  try {
    await downloadFile(releaseUrl, tmpZip);

    // Extract
    fs.mkdirSync(FRAMEWORKS_DIR, { recursive: true });
    execSync(`unzip -o -q "${tmpZip}" -d "${FRAMEWORKS_DIR}"`, { stdio: 'inherit' });

    // Cleanup
    fs.unlinkSync(tmpZip);

    log('iOS frameworks installed successfully.');
  } catch (err) {
    // Cleanup partial download
    try { fs.unlinkSync(tmpZip); } catch {}

    log(`Error: Could not download iOS frameworks: ${err.message}`);
    log('iOS builds will not work until frameworks are available.');
    log('Run: ./scripts/download-ios-frameworks.sh to download manually,');
    log('  or: ./scripts/build-ios-engine.sh to build from source.');

    // Fail fast on macOS so users discover the problem now, not at Xcode link time.
    // Skip SKIP_IOS_FRAMEWORK_DOWNLOAD is already checked above.
    if (process.platform === 'darwin') {
      log('Set SKIP_IOS_FRAMEWORK_DOWNLOAD=1 to suppress this error (e.g. Android-only builds).');
      process.exit(1);
    }
  }
}

main();
