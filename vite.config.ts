import { defineConfig } from 'vite';

// Cross-origin isolation enables multi-threaded WASM (SharedArrayBuffer) for onnxruntime-web.
// `credentialless` keeps CORS fetches to the Hugging Face / CDN model hosts working.
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  base: './',
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
