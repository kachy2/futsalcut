// Builds bench/index.html into one self-contained HTML file (opens from file://, a secure context).
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  root: 'bench',
  build: { outDir: '../dist-bench', emptyOutDir: true, target: 'es2022' },
  plugins: [viteSingleFile()],
});
