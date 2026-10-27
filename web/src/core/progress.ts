/** Time estimates shown while an analysis runs. */

/** Stage that works through `total` similar items (person detection, stored-frame replay): seconds per item and left.
 * Once two items are done the stage's own pace is used; before that the worker's a-priori guess, if it has one. */
export function itemEta(done: number, total: number, stageWallS?: number, hintS?: number): { perItemS: number; leftS: number } | null {
  const per = done >= 2 && stageWallS ? stageWallS / done : hintS;
  return per ? { perItemS: per, leftS: Math.max(0, total - done) * per } : null;
}

/** Decode stage: video seconds covered per wall second, and the wall time left for the rest of the session. */
export function decodeEta(sessionDoneS: number, sessionS: number, wallS: number): { rate: number; leftS: number } | null {
  if (!sessionDoneS) return null;
  const rate = sessionDoneS / Math.max(0.1, wallS);
  return { rate, leftS: Math.max(0, sessionS - sessionDoneS) / rate };
}
