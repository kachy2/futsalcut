/** Types shared by the UI and the workers. A session is one game made of one or more split files. */
import type { Pass1, Perspective, Scores } from './features';
import type { Point } from './roi';

export interface SessionFile {
  name: string;
  size: number;
  codec: string;
  codecString: string;
  width: number;
  height: number;
  rotation: number;
  durationS: number;
  firstTimestamp: number; // seconds; local times are relative to this
  offsetS: number;        // session time of the file's start
  supported: boolean;
  medianGopS: number;
}

export interface AnalyzeRequest {
  type: 'analyze';
  files: File[];
  zone: Point[];
  detector: 'auto' | 'off';
  /** person-detector backend: 'auto' (default) uses the graphics chip where it is real and faster, 'wasm' only the CPU, 'webgpu' insists on the GPU */
  detectorBackend?: 'auto' | 'wasm' | 'webgpu';
  /** WASM threads for the detector (default: half of the cores, at most 4, only on a cross-origin isolated page) */
  ortThreads?: number;
  /** seconds WebGPU may take to set up before the detector falls back to WASM (default 60; a software adapter needs more) */
  webgpuWaitS?: number;
  /** test override: 'auto' may also pick a software WebGPU adapter */
  anyGpu?: boolean;
  /** same files as the last analysis and only the zone changed: reuse its decoded samples and detector boxes */
  reuse?: boolean;
  modelUrl: string;
  modelJson: unknown;
}

export interface Diagnostics {
  wallS: number;
  decodeS: number;
  featuresS: number;
  detectorS: number;
  samples: number;
  sessionS: number;
  xRealtime: number;
  frameFormats: string[];
  exactPath: number;
  fallbackPath: number;
  detectorBackend: string;
  /** what was asked for: auto, wasm or webgpu */
  detectorAsked?: string;
  detectorFallbackReason?: string | null;
  /** WebGPU: how far its results were from WASM on the check frame, and in 'auto' mode the time per frame of both */
  detectorGpuCheck?: string | null;
  detectorCalls: number;
  detectorMsPerCall: number;
  /** WASM threads the detector runs with (more than 1 needs a cross-origin isolated page) */
  detectorThreads?: number;
  crossOriginIsolated?: boolean;
  /** true when the decoded samples of an earlier run were reused (decodeS is then 0) */
  reused?: boolean;
  replayS?: number;
  /** where the decode stage spent its time, in seconds: luma sample, pass 1, thumbnails, sparse detector calls */
  lumaS?: number;
  pass1S?: number;
  thumbS?: number;
  strideDetectorS?: number;
  /** creating the detector: model download, sessions, and for 'auto' / WebGPU the check against the CPU */
  detectorSetupS?: number;
  /** inside lumaS: copying the frame out of the decoder, and shrinking it to the 640-px luma */
  lumaCopyS?: number;
  lumaShrinkS?: number;
  /** drawing frames to the detector's 640x640 input and reading the pixels back (decode-stage and cascade calls together) */
  frameToBoxesS?: number;
  /** cascade stage: waiting for the key-frame decode of the next uncertain sample */
  redecodeS?: number;
  /** how the samples are kept for a new zone: file (private file system), memory or none */
  sampleStore?: string;
  perspective: Perspective;
  userAgent: string;
}

export interface AnalysisResult {
  files: SessionFile[];
  t: Float64Array;          // session time per sample
  fileIndex: Int32Array;
  localT: Float64Array;     // time within its file, relative to firstTimestamp
  scores: Scores;
  light: Float64Array;
  prob: Float64Array;       // final probabilities (cascade)
  detector: [number, number][]; // [sample index, detector score]
  perspective: Perspective;
  diagnostics: Diagnostics;
}

export type WorkerMessage =
  | { type: 'files'; files: SessionFile[] }
  | { type: 'progress'; stage: 'decode' | 'replay' | 'detector' | 'finalize'; done: number; total: number; sessionDoneS: number; wallS: number;
      /** seconds since this stage began, and an a-priori seconds-per-item guess for the first message of a stage */
      stageWallS?: number; itemHintS?: number;
      /** decode stage: rough seconds the person detection will take afterwards (about 30% of the samples are uncertain) */
      detGuessS?: number }
  | { type: 'thumbs'; start: number; blobs: Blob[] }
  | { type: 'result'; result: AnalysisResult }
  | { type: 'cancelled' }
  | { type: 'error'; message: string };

export type { Pass1 };

/** Split session intervals into per-file local segments for the exporter. */
export function toFileSegments(intervals: [number, number][], files: SessionFile[]): { file: number; start: number; end: number }[] {
  const out: { file: number; start: number; end: number }[] = [];
  for (const [a, b] of intervals) {
    files.forEach((f, k) => {
      const s = Math.max(a, f.offsetS);
      const e = Math.min(b, f.offsetS + f.durationS);
      if (e - s > 1e-3) out.push({ file: k, start: s - f.offsetS, end: e - f.offsetS });
    });
  }
  return out;
}
