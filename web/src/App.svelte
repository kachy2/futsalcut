<script lang="ts">
  import { calibrate, pickQuestions, quickCalibrate } from './core/ensemble';
  import type { Scores } from './core/features';
  import { postParams, type ModelFile } from './core/model';
  import type { Point } from './core/roi';
  import { keptSeconds, select, thresholdForBudget, type Interval, type PostParams } from './core/select';
  import { toFileSegments, type AnalysisResult, type SessionFile, type WorkerMessage } from './core/session';
  import FilePicker from './ui/FilePicker.svelte';
  import Guide from './ui/Guide.svelte';
  import QuickCalibration from './ui/QuickCalibration.svelte';
  import RoiEditor from './ui/RoiEditor.svelte';
  import Timeline from './ui/Timeline.svelte';
  import { strings, type Lang } from './ui/i18n';

  const store = {
    get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  // English unless the link says ?lang=ko (or en) or the visitor picked a language before
  const isLang = (v: string | null): v is Lang => v === 'ko' || v === 'en';
  const urlLang = new URLSearchParams(location.search).get('lang');
  if (isLang(urlLang)) store.set('lang', urlLang);
  const savedLang = store.get('lang');
  let lang = $state<Lang>(isLang(urlLang) ? urlLang : isLang(savedLang) ? savedLang : 'en');
  const t = $derived(strings[lang]);
  // WebCodecs (and the save dialog) exist only in secure contexts: https or localhost
  const envError = $derived(typeof VideoDecoder !== 'undefined' ? '' : isSecureContext ? t.errNoWebCodecs : t.errInsecure);
  $effect(() => {
    document.title = t.docTitle;
    document.documentElement.lang = lang;
  });

  let step = $state(1);
  let files = $state<File[]>([]);
  let bitmap = $state<ImageBitmap | null>(null);
  let zone = $state<Point[]>([]);
  let model = $state<ModelFile | null>(null);
  let error = $state('');
  let sessionFiles = $state<SessionFile[]>([]);
  let progress = $state<{ stage: string; done: number; total: number; sessionDoneS: number; wallS: number } | null>(null);
  let result = $state.raw<AnalysisResult | null>(null);
  let thumbs = $state.raw<Blob[]>([]);

  // controls
  let mode = $state<'auto' | 'sens' | 'target'>('auto');
  let sensitivity = $state(70);
  let targetMin = $state(10);
  let algo = $state('auto');
  let showAdvanced = $state(false);
  let showLanes = $state(false);
  let post = $state<PostParams | null>(null);
  let qcQuestions = $state<number[] | null>(null);

  // export
  let exporting = $state<{ doneS: number; totalS: number } | null>(null);
  let exportDone = $state<{ bytes: number; durationS: number; wallS: number } | null>(null);
  let copied = $state(false);
  let probe = $state<string>('');

  function runProbe() {
    probe = '…';
    const w = new Worker(new URL('./worker/detector-probe.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e) => { probe = JSON.stringify(e.data); w.terminate(); };
    w.onerror = (e) => { probe = 'error: ' + e.message; w.terminate(); };
    w.postMessage({ modelUrl: new URL('models/dfine_n_coco.onnx', base).href });
  }

  const base = document.baseURI;

  // keep the screen awake during long analysis/export (a sleeping tab gets throttled)
  let wake: { release(): Promise<void> } | null = null;
  async function stayAwake(on: boolean) {
    try {
      if (on && !wake) wake = await (navigator as unknown as { wakeLock?: { request(t: string): Promise<{ release(): Promise<void> }> } }).wakeLock?.request('screen') ?? null;
      if (!on && wake) { await wake.release(); wake = null; }
    } catch { /* not supported or not allowed */ }
  }
  $effect(() => { stayAwake(!!progress || !!exporting); });
  fetch(new URL('models/ensemble_v1.json', base)).then((r) => r.json()).then((m: ModelFile) => {
    model = m;
    post = postParams(m);
  }).catch((e) => (error = String(e)));

  const analyzer = new Worker(new URL('./worker/analyze.ts', import.meta.url), { type: 'module' });
  analyzer.onmessage = (ev: MessageEvent) => {
    const m = ev.data as WorkerMessage | { type: 'preview'; bitmap: ImageBitmap };
    if (m.type === 'preview') bitmap = m.bitmap;
    else if (m.type === 'files') sessionFiles = m.files;
    else if (m.type === 'progress') progress = m;
    else if (m.type === 'thumbs') {
      const next = thumbs.slice();
      m.blobs.forEach((b, k) => (next[m.start + k] = b));
      thumbs = next;
    } else if (m.type === 'result') {
      result = m.result;
      progress = null;
      step = 4;
    } else if (m.type === 'error') {
      error = m.message;
      progress = null;
    }
  };
  const exporter = new Worker(new URL('./worker/export.ts', import.meta.url), { type: 'module' });
  exporter.onmessage = (ev: MessageEvent) => {
    const m = ev.data;
    if (m.type === 'progress') exporting = { doneS: m.doneS, totalS: m.totalS };
    else if (m.type === 'done') {
      exporting = null;
      exportDone = { bytes: m.check.bytes, durationS: m.check.durationS, wallS: m.report.wallS };
    } else if (m.type === 'error') {
      exporting = null;
      error = m.message;
    }
  };

  function toStep2() {
    error = '';
    bitmap = null;
    step = 2;
    analyzer.postMessage({ type: 'preview', file: files[0], atRatio: 0.1 });
  }
  function startAnalysis() {
    error = '';
    thumbs = [];
    result = null;
    exportDone = null;
    step = 3;
    progress = { stage: 'decode', done: 0, total: 0, sessionDoneS: 0, wallS: 0 };
    analyzer.postMessage({
      type: 'analyze', files: $state.snapshot(files), zone: $state.snapshot(zone), detector: 'auto',
      modelUrl: new URL('models/dfine_n_coco.onnx', base).href,
      modelJson: $state.snapshot(model),
    });
  }

  // probabilities available for selection
  const lightNames = $derived(model?.light.names ?? []);
  const probs = $derived.by(() => {
    const out: Record<string, Float64Array> = {};
    if (!result || !model) return out;
    out.auto = result.prob;
    out.light = result.light;
    for (const name of lightNames) {
      const c = model.light.calibrators[name];
      const s = (result.scores as unknown as Record<string, Float64Array>)[name];
      out[name] = Float64Array.from(s, (v) => calibrate(c, v));
    }
    return out;
  });
  const prob = $derived(probs[algo] ?? probs.auto ?? new Float64Array(0));
  const totalS = $derived(result ? result.files.reduce((s, f) => s + f.durationS, 0) : 0);
  const selection = $derived.by((): { threshold: number | null; intervals: Interval[] } => {
    if (!result || !model || !post) return { threshold: null, intervals: [] };
    if (mode === 'target') {
      const r = thresholdForBudget(prob, result.t, targetMin * 60, post, totalS);
      return { threshold: r.threshold, intervals: r.intervals };
    }
    const thr = mode === 'auto' && algo === 'auto' ? model.threshold : Math.min(0.98, Math.max(0.02, 1 - sensitivity / 100));
    return { threshold: thr, intervals: select(prob, result.t, thr, post, totalS) };
  });
  const keptS = $derived(keptSeconds(selection.intervals));
  // read-only hook for automated tests and power users (no effect on the app)
  $effect(() => {
    (window as unknown as { __futsalcut?: unknown }).__futsalcut = {
      intervals: selection.intervals, threshold: selection.threshold, t: result ? Array.from(result.t) : [],
      prob: Array.from(prob), light: result ? Array.from(result.light) : [], detector: result?.detector ?? [],
    };
  });

  function openQuickCal() {
    if (!result) return;
    const p: Record<string, Float64Array> = { ensemble: probs.auto };
    for (const n of lightNames) p[n] = probs[n];
    qcQuestions = pickQuestions(p, result.t, 10, 20);
  }
  function applyQuickCal(answers: Map<number, boolean>) {
    const p: Record<string, Float64Array> = { ensemble: probs.auto };
    for (const n of lightNames) p[n] = probs[n];
    const r = quickCalibrate(p, answers, 'ensemble');
    algo = r.name === 'ensemble' ? 'auto' : r.name;
    mode = 'sens';
    sensitivity = Math.round((1 - r.threshold) * 100);
    qcQuestions = null;
  }

  async function save() {
    if (!result) return;
    const picker = (window as unknown as { showSaveFilePicker?: (o: unknown) => Promise<FileSystemFileHandle> }).showSaveFilePicker;
    if (!picker) { error = t.noFsa; return; }
    error = '';
    exportDone = null;
    const stem = files[0].name.replace(/\.[^.]+$/, '');
    let handle: FileSystemFileHandle;
    try {
      handle = await picker({ suggestedName: `${stem}_play.mp4`, types: [{ description: 'MP4', accept: { 'video/mp4': ['.mp4'] } }] });
    } catch { return; } // user cancelled
    exporting = { doneS: 0, totalS: keptS };
    exporter.postMessage({ type: 'export', files: $state.snapshot(files), segments: toFileSegments(selection.intervals, result.files), handle });
  }

  const fmt = (s: number) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${m}:${String(sec).padStart(2, '0')}`;
  };
  const sessionS = $derived(sessionFiles.reduce((s, f) => s + f.durationS, 0));
  const eta = $derived.by(() => {
    if (!progress || progress.stage !== 'decode' || !progress.sessionDoneS) return null;
    const rate = progress.sessionDoneS / Math.max(0.1, progress.wallS);
    return { rate, left: (sessionS - progress.sessionDoneS) / rate };
  });
</script>

<main>
  <div class="row">
    <div>
      <h1>{t.title}</h1>
      <p class="muted" style="margin:2px 0 0">{t.tagline}</p>
    </div>
    <span class="spacer"></span>
    <button onclick={() => { lang = lang === 'ko' ? 'en' : 'ko'; store.set('lang', lang); }}>{t.lang}</button>
  </div>
  <p class="small muted" style="margin-top:6px">🔒 {t.privacy}</p>
  {#if envError}<div class="card error">{envError}</div>{/if}
  {#if error}<div class="card error">{error}</div>{/if}

  {#if step === 1}
    <div class="card">
      <h2>{t.step1}</h2>
      <FilePicker bind:files {t} />
      <div class="row" style="margin-top:12px"><span class="spacer"></span>
        <button class="primary" disabled={!files.length || !!envError} onclick={toStep2}>{t.next}</button></div>
    </div>
    <Guide {t} />
  {:else if step === 2}
    <div class="card">
      <h2>{t.step2}</h2>
      <RoiEditor {bitmap} bind:zone {t} />
      <div class="row" style="margin-top:12px">
        <button onclick={() => (step = 1)}>{t.back}</button><span class="spacer"></span>
        <button class="primary" disabled={zone.length < 3 || !model} onclick={startAnalysis}>{t.analyze}</button>
      </div>
    </div>
  {:else if step === 3}
    <div class="card">
      <h2>{t.step3}</h2>
      {#if progress}
        <p>{progress.stage === 'detector' ? t.detecting : t.decoding}</p>
        <div class="bar"><div style="width:{progress.stage === 'detector'
          ? (100 * progress.done) / Math.max(1, progress.total)
          : (100 * progress.sessionDoneS) / Math.max(1, sessionS)}%"></div></div>
        <p class="small muted" style="margin-top:6px">
          {fmt(progress.sessionDoneS)} / {fmt(sessionS)}
          {#if eta} · {t.speed} {eta.rate.toFixed(1)}{t.realtime} · {t.eta} {fmt(eta.left)}{/if}
        </p>
        <button onclick={() => analyzer.postMessage({ type: 'cancel' })}>{t.cancel}</button>
      {/if}
    </div>
  {:else if step >= 4 && result}
    <div class="card">
      <div class="row">
        <h2 style="margin:0">{t.step4}</h2><span class="spacer"></span>
        <span class="stat">{fmt(keptS)}</span>
        <span class="muted">/ {fmt(totalS)} · {selection.intervals.length} {t.scenes} · {((100 * keptS) / Math.max(1, totalS)).toFixed(1)}%</span>
      </div>
      <Timeline times={result.t} {prob} threshold={mode === 'target' ? null : selection.threshold}
        intervals={selection.intervals} total={totalS} boundaries={result.files.slice(1).map((f) => f.offsetS)}
        lanes={showLanes ? lightNames.map((n) => ({ name: n, values: probs[n] })) : []} {thumbs} />
      <div class="row" style="margin-top:12px">
        <label>{t.mode}
          <select bind:value={mode}>
            <option value="auto">{t.modeAuto}</option>
            <option value="sens">{t.modeSens}</option>
            <option value="target">{t.modeTarget}</option>
          </select>
        </label>
        {#if mode === 'sens'}
          <span class="small muted">{t.lessKeep}</span>
          <input type="range" min="2" max="98" bind:value={sensitivity} />
          <span class="small muted">{t.moreKeep}</span>
        {:else if mode === 'target'}
          <label>{t.targetLen} <input type="number" min="1" max={Math.ceil(totalS / 60)} bind:value={targetMin} style="width:70px" /></label>
        {/if}
        <span class="spacer"></span>
        <button onclick={openQuickCal}>{t.quickCal}</button>
        <button onclick={() => (showAdvanced = !showAdvanced)}>{t.advanced}</button>
      </div>
      {#if showAdvanced && post}
        <div class="row small" style="margin-top:10px">
          <label>{t.algorithm}
            <select bind:value={algo}>
              <option value="auto">{t.algoAuto}</option>
              <option value="light">{t.algoLight}</option>
              {#each lightNames as n}<option value={n}>{n}</option>{/each}
            </select>
          </label>
          <label>{t.minScene} <input type="number" step="1" min="1" bind:value={post.minLen} style="width:60px" /></label>
          <label title={t.padBeforeHint}>{t.padBefore} <input type="number" step="1" min="0" max="30" bind:value={post.padBefore} style="width:60px" /></label>
          <label>{t.padAfter} <input type="number" step="1" min="0" max="30" bind:value={post.padAfter} style="width:60px" /></label>
          <label>{t.mergeGap} <input type="number" step="1" min="0" bind:value={post.mergeGap} style="width:60px" /></label>
          <label><input type="checkbox" bind:checked={showLanes} /> {t.lanes}</label>
        </div>
      {/if}
      {#if !selection.intervals.length}<p class="error small" style="margin-top:8px">{t.noScenes}</p>{/if}
    </div>

    {#if qcQuestions}
      <QuickCalibration questions={qcQuestions} {thumbs} times={result.t} {t} onapply={applyQuickCal} oncancel={() => (qcQuestions = null)} />
    {/if}

    <div class="card">
      <h2>{t.step5}</h2>
      <div class="row">
        <button class="primary" disabled={!selection.intervals.length || !!exporting} onclick={save}>{t.save}</button>
        <button onclick={() => { step = 1; files = []; zone = []; result = null; thumbs = []; }}>{t.startOver}</button>
      </div>
      {#if exporting}
        <p class="small" style="margin-top:8px">{t.saving} {fmt(exporting.doneS)} / {fmt(exporting.totalS)}</p>
        <div class="bar"><div style="width:{(100 * exporting.doneS) / Math.max(1, exporting.totalS)}%"></div></div>
      {/if}
      {#if exportDone}
        <p style="margin-top:8px">✅ {t.saved}: {fmt(exportDone.durationS)} · {(exportDone.bytes / 2 ** 30).toFixed(2)} GB · {exportDone.wallS.toFixed(1)} s</p>
        <p class="small muted">{t.savedInfo}</p>
      {/if}
    </div>

    <details class="card">
      <summary>{t.diagnostics}</summary>
      <pre>{JSON.stringify({ ...result.diagnostics, files: result.files }, null, 1)}</pre>
      <div class="row">
        <button onclick={async () => { await navigator.clipboard.writeText(JSON.stringify({ ...result!.diagnostics, files: result!.files }, null, 1)); copied = true; }}>
          {copied ? t.copied : t.copy}</button>
        <button id="probe" onclick={runProbe}>detector test</button>
      </div>
      {#if probe}<pre id="probe-result">{probe}</pre>{/if}
    </details>
  {/if}
  <p class="small muted footer">
    FutsalCut · MIT ·
    <a href="https://github.com/kachy2/futsalcut" target="_blank" rel="noopener">GitHub</a> ·
    <a href="https://github.com/kachy2/futsalcut/issues/new/choose" target="_blank" rel="noopener">{t.feedbackLink}</a>
  </p>
</main>
