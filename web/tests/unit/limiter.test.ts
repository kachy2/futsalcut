import { describe, expect, it } from 'vitest';
import { Limiter } from '../../src/core/limiter';

/** A task that ends when `finish` is called. */
function gate() {
  let finish!: () => void;
  let fail!: (e: Error) => void;
  const done = new Promise<void>((resolve, reject) => { finish = resolve; fail = reject; });
  return { done, finish, fail };
}
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

describe('Limiter', () => {
  it('starts a task at once when a slot is free and never has more than cap running', async () => {
    const lim = new Limiter(2);
    const gates = [gate(), gate(), gate()];
    let running = 0;
    let peak = 0;
    const started: number[] = [];
    const task = (k: number) => async () => {
      started.push(k);
      peak = Math.max(peak, ++running);
      await gates[k].done;
      running--;
    };
    await lim.run(task(0));
    await lim.run(task(1));
    expect(started).toEqual([0, 1]); // the first two did not wait
    let third = false;
    const pending = lim.run(task(2)).then(() => { third = true; });
    await tick();
    expect(third).toBe(false); // both slots are taken
    gates[0].finish();
    await pending;
    expect(started).toEqual([0, 1, 2]);
    gates[1].finish();
    gates[2].finish();
    await lim.drain();
    expect(peak).toBe(2);
    expect(running).toBe(0);
    expect(lim.running).toBe(0);
  });

  it('drain waits for every task, also for the ones started before', async () => {
    const lim = new Limiter(3);
    const g = [gate(), gate()];
    const ended: number[] = [];
    for (let k = 0; k < 2; k++) await lim.run(async () => { await g[k].done; ended.push(k); });
    let drained = false;
    const d = lim.drain().then(() => { drained = true; });
    g[1].finish();
    await tick();
    expect(drained).toBe(false);
    g[0].finish();
    await d;
    expect(ended).toEqual([1, 0]);
  });

  it('keeps the first failure for the next run and for drain, and lets the other tasks finish', async () => {
    const lim = new Limiter(3);
    const a = gate();
    const b = gate();
    let finished = false;
    await lim.run(async () => { await a.done; });
    await lim.run(async () => { await b.done; finished = true; });
    a.fail(new Error('boom'));
    await tick();
    await expect(lim.run(async () => undefined)).rejects.toThrow('boom'); // no new task after a failure
    b.finish();
    await expect(lim.drain()).rejects.toThrow('boom');
    expect(finished).toBe(true);
  });

  it('turns a synchronous throw into a failure and keeps the first of several', async () => {
    const lim = new Limiter(2);
    await lim.run(() => { throw new Error('first'); });
    await expect(lim.drain()).rejects.toThrow('first');
    const other = new Limiter(2);
    const g = gate();
    await other.run(async () => { await g.done; });
    await other.run(async () => { throw new Error('early'); });
    await tick(); // the early failure is recorded before the late one happens
    g.fail(new Error('late'));
    await expect(other.drain()).rejects.toThrow('early');
  });

  it('a waiting run is released by a failure instead of waiting for a slot forever', async () => {
    const lim = new Limiter(1);
    const g = gate();
    await lim.run(async () => { await g.done; });
    const waiting = lim.run(async () => undefined);
    g.fail(new Error('gone'));
    await expect(waiting).rejects.toThrow('gone');
  });
});
