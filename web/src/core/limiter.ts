/**
 * Runs async tasks with at most `cap` of them in flight. `run` waits for a free slot, starts the task and returns while it goes on;
 * `drain` waits for the rest. The first failure of any task is kept and thrown by the next `run` and by `drain`, so a task that
 * nobody awaits cannot fail unnoticed. Tasks start in the order they are given.
 */
export class Limiter {
  private active = new Set<Promise<void>>();
  private failure: { error: unknown } | null = null;

  constructor(private readonly cap: number) {}

  /** tasks in flight right now */
  get running(): number { return this.active.size; }

  /** `run` would start a task at once */
  get hasRoom(): boolean { return this.active.size < this.cap && !this.failure; }

  /** a task has failed; `run` and `drain` throw that failure */
  get failed(): boolean { return !!this.failure; }

  /** Resolves when a task settles, or at once if there is room (or a failure): for choosing between several limiters. */
  async untilRoom(): Promise<void> {
    if (this.active.size >= this.cap && !this.failure) await Promise.race(this.active);
  }

  async run(task: () => Promise<void>): Promise<void> {
    while (this.active.size >= this.cap && !this.failure) await Promise.race(this.active);
    if (this.failure) throw this.failure.error;
    const p: Promise<void> = new Promise<void>((resolve) => resolve(task()))
      .catch((error) => { this.failure ??= { error }; })
      .finally(() => { this.active.delete(p); });
    this.active.add(p);
  }

  async drain(): Promise<void> {
    while (this.active.size) await Promise.race(this.active);
    if (this.failure) throw this.failure.error;
  }
}
