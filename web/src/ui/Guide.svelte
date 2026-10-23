<script lang="ts">
  import type { Strings } from './i18n';

  let { t }: { t: Strings } = $props();
</script>

<section class="card guide">
  <h2>{t.guideTitle}</h2>
  <ol>{#each t.guideSteps as s}<li>{s}</li>{/each}</ol>
  <details>
    <summary>{t.tipsTitle}</summary>
    <ul>{#each t.tips as s}<li>{s}</li>{/each}</ul>
  </details>
  <details>
    <summary>{t.faqTitle}</summary>
    {#each t.faq as f}<p><strong>{f.q}</strong><br />{f.a}</p>{/each}
  </details>
</section>
