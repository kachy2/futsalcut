<script lang="ts">
  import type { Strings } from './i18n';

  let { files = $bindable([]), t }: { files: File[]; t: Strings } = $props();
  let over = $state(false);
  let input: HTMLInputElement;

  function accept(list: FileList | null) {
    if (!list) return;
    const vids = [...list].filter((f) => /\.(mp4|mov|m4v)$/i.test(f.name) || f.type.startsWith('video/'));
    files = vids.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  }
  const gb = (n: number) => (n / 2 ** 30).toFixed(2) + ' GB';
</script>

<div
  class="drop {over ? 'over' : ''}"
  role="button"
  tabindex="0"
  onclick={() => input.click()}
  onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && input.click()}
  ondragover={(e) => { e.preventDefault(); over = true; }}
  ondragleave={() => (over = false)}
  ondrop={(e) => { e.preventDefault(); over = false; accept(e.dataTransfer?.files ?? null); }}
>
  {t.dropZone}
  <input bind:this={input} type="file" accept="video/*,.mp4,.mov,.m4v" multiple hidden
    onchange={(e) => accept((e.currentTarget as HTMLInputElement).files)} />
</div>
{#if files.length}
  <table style="margin-top:10px">
    <thead><tr><th>#</th><th>{t.files}</th><th>size</th></tr></thead>
    <tbody>
      {#each files as f, i}
        <tr><td>{i + 1}</td><td>{f.name}</td><td>{gb(f.size)}</td></tr>
      {/each}
    </tbody>
  </table>
{/if}
