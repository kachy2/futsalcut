import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'node:path';
import { defaultClientConditions, defineConfig } from 'vite';

// GitHub Pages serves the app from /<repo>/; override with BASE_URL for other hosts.
export default defineConfig({
  base: process.env.BASE_URL ?? './',
  plugins: [svelte()],
  worker: { format: 'es' },
  // onnxruntime-web: load its runtime (.mjs + .wasm, copied to public/ort/ by scripts/prepare-assets.mjs) as separate files;
  // the bundled variant cannot start its WASM threads from inside our worker chunk
  resolve: { conditions: [...defaultClientConditions, 'onnxruntime-web-use-extern-wasm'] },
  // two entry pages share one bundle: the English app at / and a Korean landing page at /ko/ (own title, description, canonical)
  build: {
    target: 'es2022', outDir: 'dist', emptyOutDir: true,
    rollupOptions: { input: { main: resolve(__dirname, 'index.html'), ko: resolve(__dirname, 'ko/index.html') } },
  },
  server: { host: '0.0.0.0', port: 5173 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
