// Copy the shipped models into public/models and the ONNX Runtime files into public/ort.
// The detector model is exported by eval/runners/export_dfine.py (Apache-2.0, see public/models/NOTICE.md).
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const models = resolve(root, 'public/models');
mkdirSync(models, { recursive: true });
copyFileSync(resolve(root, '../python/src/futsalcut/models/ensemble_v1.json'), resolve(models, 'ensemble_v1.json'));
const onnx = process.env.DFINE_ONNX ?? '/data/models/dfine_n_coco_web.onnx'; // constant-folded for onnxruntime-web
if (existsSync(onnx)) copyFileSync(onnx, resolve(models, 'dfine_n_coco.onnx')); // otherwise the committed copy is used
if (!existsSync(resolve(models, 'dfine_n_coco.onnx'))) console.warn('detector model missing: set DFINE_ONNX');
// onnxruntime-web runtime, served as plain files at <app>/ort/ (see vite.config.ts); not committed (26 MB)
const ort = resolve(root, 'public/ort');
mkdirSync(ort, { recursive: true });
for (const f of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  copyFileSync(resolve(root, 'node_modules/onnxruntime-web/dist', f), resolve(ort, f));
}
console.log('assets ready:', readdirSync(models).join(', '), '| ort:', readdirSync(ort).join(', '));
