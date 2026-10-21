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
  detectorCalls: number;
  detectorMsPerCall: number;
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
  | { type: 'progress'; stage: 'decode' | 'detector' | 'finalize'; done: number; total: number; sessionDoneS: number; wallS: number }
  | { type: 'thumbs'; start: number; blobs: Blob[] }
  | { type: 'result'; result: AnalysisResult }
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
