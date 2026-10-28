<script lang="ts">
  import type { Strings } from './i18n';

  let { questions, thumbs, times, t, onapply, oncancel }: {
    questions: number[]; thumbs: Blob[]; times: Float64Array; t: Strings;
    onapply: (answers: Map<number, boolean>) => void; oncancel: () => void;
  } = $props();

  let answers = $state<Record<number, boolean | undefined>>({});
  const urls = $derived(questions.map((i) => (thumbs[i] ? URL.createObjectURL(thumbs[i]) : '')));
  $effect(() => () => urls.forEach((u) => u && URL.revokeObjectURL(u)));
  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const answered = $derived(questions.filter((i) => answers[i] !== undefined).length);
  const done = $derived(answered === questions.length);
</script>

<div class="card">
  <h2>{t.quickCal}</h2>
  <p class="muted small">{t.quickCalHelp}</p>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px">
    {#each questions as i, k}
      <div style="border:1px solid var(--line);border-radius:8px;padding:6px">
        {#if urls[k]}<img src={urls[k]} alt="" style="width:100%;border-radius:4px;display:block" />{/if}
        <div class="row small" style="margin-top:4px">
          <span>{fmt(times[i])}</span><span class="spacer"></span>
          <button class={answers[i] === true ? 'primary' : ''} onclick={() => (answers[i] = true)}>{t.keep}</button>
          <button class={answers[i] === false ? 'primary' : ''} onclick={() => (answers[i] = false)}>{t.drop}</button>
        </div>
      </div>
    {/each}
  </div>
  <div class="row" style="margin-top:10px">
    <button class="primary" disabled={!done}
      onclick={() => onapply(new Map(questions.map((i) => [i, answers[i]!] as [number, boolean])))}>{t.apply}</button>
    <button onclick={oncancel}>{t.cancel}</button>
    <span class="small muted">{answered} / {questions.length}</span>
  </div>
</div>
