<script lang="ts">
  import type { Interval } from '../core/select';

  let {
    times, prob, threshold, intervals, total, boundaries = [], lanes = [], thumbs = [],
  }: {
    times: Float64Array; prob: Float64Array; threshold: number | null; intervals: Interval[]; total: number;
    boundaries?: number[]; lanes?: { name: string; values: ArrayLike<number> }[]; thumbs?: Blob[];
  } = $props();

  let wrap: HTMLDivElement;
  let canvas: HTMLCanvasElement;
  let width = $state(800);
  let hover = $state<{ i: number; x: number } | null>(null);
  const urls = new Map<number, string>();
  const MAIN_H = 110;
  const LANE_H = 20;
  const AXIS_H = 18;

  const fmt = (s: number) => {
    const m = Math.floor(s / 60);
    return `${Math.floor(m / 60) ? Math.floor(m / 60) + ':' + String(m % 60).padStart(2, '0') : m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  };

  $effect(() => {
    const ro = new ResizeObserver(() => (width = wrap.clientWidth));
    ro.observe(wrap);
    return () => ro.disconnect();
  });

  function css(name: string) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
  }

  $effect(() => {
    void [times, prob, threshold, intervals, total, lanes, width];
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const H = MAIN_H + lanes.length * LANE_H + AXIS_H;
    canvas.width = width * dpr;
    canvas.height = H * dpr;
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, H);
    const X = (s: number) => (s / Math.max(1e-9, total)) * width;
    // kept intervals
    ctx.fillStyle = css('--keep');
    ctx.globalAlpha = 0.22;
    for (const [a, b] of intervals) ctx.fillRect(X(a), 0, Math.max(1, X(b) - X(a)), MAIN_H);
    ctx.globalAlpha = 1;
    for (const [a, b] of intervals) ctx.fillRect(X(a), MAIN_H - 8, Math.max(1, X(b) - X(a)), 8);
    // probability
    ctx.strokeStyle = css('--prob');
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < times.length; i++) {
      const x = X(times[i] + 0.5);
      const y = (MAIN_H - 12) * (1 - prob[i]) + 2;
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    }
    ctx.stroke();
    if (threshold !== null) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = css('--muted');
      const y = (MAIN_H - 12) * (1 - threshold) + 2;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // file boundaries
    ctx.strokeStyle = css('--line');
    for (const b of boundaries) {
      ctx.beginPath();
      ctx.moveTo(X(b), 0);
      ctx.lineTo(X(b), MAIN_H);
      ctx.stroke();
    }
    // lanes
    lanes.forEach((lane, k) => {
      const y0 = MAIN_H + k * LANE_H;
      for (let i = 0; i < times.length; i++) {
        const v = Math.max(0, Math.min(1, lane.values[i]));
        ctx.fillStyle = `rgba(209,73,91,${v.toFixed(3)})`;
        const x0 = X(times[i]);
        ctx.fillRect(x0, y0 + 3, Math.max(1, X((times[i + 1] ?? times[i] + 1)) - x0), LANE_H - 6);
      }
      ctx.fillStyle = css('--fg');
      ctx.font = '11px system-ui';
      ctx.fillText(lane.name, 4, y0 + LANE_H - 7);
    });
    // axis
    ctx.fillStyle = css('--muted');
    ctx.font = '11px system-ui';
    const step = total > 5400 ? 900 : total > 1800 ? 300 : total > 600 ? 120 : 60;
    for (let s = 0; s <= total; s += step) ctx.fillText(fmt(s), Math.min(width - 30, X(s) + 2), H - 4);
  });

  function onMove(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    const s = ((e.clientX - r.left) / r.width) * total;
    let lo = 0;
    let hi = times.length - 1;
    while (lo < hi) {
      const m = (lo + hi + 1) >> 1;
      if (times[m] <= s) lo = m; else hi = m - 1;
    }
    hover = { i: lo, x: e.clientX - r.left };
  }
  function thumbUrl(i: number): string | null {
    if (!thumbs[i]) return null;
    let u = urls.get(i);
    if (!u) {
      u = URL.createObjectURL(thumbs[i]);
      urls.set(i, u);
    }
    return u;
  }
  $effect(() => () => { for (const u of urls.values()) URL.revokeObjectURL(u); });
</script>

<div bind:this={wrap} style="position:relative">
  <canvas bind:this={canvas} style="width:100%;display:block" onpointermove={onMove} onpointerleave={() => (hover = null)}></canvas>
  {#if hover}
    {@const url = thumbUrl(hover.i)}
    <div style="position:absolute;top:-130px;left:{Math.min(Math.max(0, hover.x - 100), width - 200)}px;width:200px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:4px;pointer-events:none;z-index:2">
      {#if url}<img src={url} alt="" style="width:100%;display:block;border-radius:4px" />{/if}
      <div class="small">{fmt(times[hover.i])} · p={prob[hover.i].toFixed(2)}</div>
    </div>
  {/if}
</div>
