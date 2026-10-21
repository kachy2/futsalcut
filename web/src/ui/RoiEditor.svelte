<script lang="ts">
  import type { Point } from '../core/roi';
  import type { Strings } from './i18n';

  let { bitmap, zone = $bindable([]), t }: { bitmap: ImageBitmap | null; zone: Point[]; t: Strings } = $props();
  let canvas = $state<HTMLCanvasElement>();
  let drag = -1;

  // default suggestion: lower 75% of the frame, slightly narrower at the top (camera-side half)
  const suggestion: Point[] = [[0.12, 0.25], [0.88, 0.25], [1, 1], [0, 1]];

  function draw(c: HTMLCanvasElement, bmp: ImageBitmap, pts: Point[]) {
    const ctx = c.getContext('2d')!;
    c.width = bmp.width;
    c.height = bmp.height;
    ctx.drawImage(bmp, 0, 0);
    if (!pts.length) return;
    ctx.fillStyle = 'rgba(46,139,87,0.25)';
    ctx.strokeStyle = '#ffd400';
    ctx.lineWidth = Math.max(2, c.width / 400);
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * c.width, y * c.height) : ctx.moveTo(x * c.width, y * c.height)));
    ctx.closePath();
    if (pts.length >= 3) ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#ff3b30';
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x * c.width, y * c.height, Math.max(6, c.width / 160), 0, Math.PI * 2);
      ctx.fill();
    }
  }
  $effect(() => {
    if (canvas && bitmap) draw(canvas, bitmap, zone);
  });

  function pos(c: HTMLCanvasElement, e: PointerEvent): Point {
    const r = c.getBoundingClientRect();
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  }
  function hit(c: HTMLCanvasElement, p: Point): number {
    const r = c.getBoundingClientRect();
    const tol = 14 / r.width;
    return zone.findIndex(([x, y]) => Math.hypot(x - p[0], (y - p[1]) * (r.height / r.width)) < tol);
  }
  function down(e: PointerEvent) {
    if (!canvas) return;
    const p = pos(canvas, e);
    const i = hit(canvas, p);
    if (e.button === 2) {
      if (i >= 0) zone = zone.filter((_, k) => k !== i);
      return;
    }
    if (i >= 0) drag = i;
    else zone = [...zone, p];
    canvas.setPointerCapture(e.pointerId);
  }
  function move(e: PointerEvent) {
    if (drag < 0 || !canvas) return;
    const p = pos(canvas, e);
    zone = zone.map((q, k) => (k === drag ? p : q));
  }
</script>

<p class="muted small">{t.roiHelp}</p>
<div class="row" style="margin-bottom:8px">
  <button onclick={() => (zone = suggestion.map((p) => [...p] as Point))}>⬚ default</button>
  <button onclick={() => (zone = [])}>{t.reset}</button>
  {#if zone.length < 3}<span class="error small">{t.roiNeed}</span>{/if}
</div>
{#if bitmap}
  <canvas bind:this={canvas} style="width:100%;touch-action:none;border-radius:8px;cursor:crosshair"
    onpointerdown={down} onpointermove={move} onpointerup={() => (drag = -1)}
    oncontextmenu={(e) => e.preventDefault()}></canvas>
{:else}
  <p class="muted">…</p>
{/if}
