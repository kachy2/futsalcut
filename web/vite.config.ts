import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vite';

// GitHub Pages serves the app from /<repo>/; override with BASE_URL for other hosts.
export default defineConfig({
  base: process.env.BASE_URL ?? './',
  plugins: [svelte()],
  worker: { format: 'es' },
  build: { target: 'es2022', outDir: 'dist', emptyOutDir: true },
  server: { host: '0.0.0.0', port: 5173 },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
