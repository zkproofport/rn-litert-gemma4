const path = require('path');
const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

// Allow Metro to follow the file: symlinks for our two workspace packages
// (we install both via `npm install file:../<pkg>`):
//   node_modules/@zkproofport/rn-on-device-mcp  → ~/Workspace/rn-on-device-mcp
//   node_modules/@zkproofport/rn-litert-gemma4  → ~/Workspace/rn-litert-gemma4
// Without watchFolders, Metro refuses to bundle code outside the project root.
const sdkRoots = [
  path.resolve(__dirname, '..', 'rn-on-device-mcp'),
  path.resolve(__dirname, '..', 'rn-litert-gemma4'),
];

/** @type {import('@react-native/metro-config').MetroConfig} */
const config = {
  watchFolders: sdkRoots,
  resolver: {
    nodeModulesPaths: [
      path.resolve(__dirname, 'node_modules'),
      ...sdkRoots.map((r) => path.join(r, 'node_modules')),
    ],
    unstable_enableSymlinks: true,
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
