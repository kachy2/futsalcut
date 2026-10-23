// Renders public/og-image.png (the 1200x630 social preview) from inline HTML; no photos, so no real people.
// Run in the web container: node scripts/make-og.mjs   (needs Playwright + Google Chrome)
import { chromium } from '@playwright/test';
import { resolve } from 'node:path';

const out = resolve(import.meta.dirname, '../public/og-image.png');
const html = `<!doctype html><meta charset="utf-8"><style>
  body { margin: 0; width: 1200px; height: 630px; position: relative; overflow: hidden; color: #fff;
    background: linear-gradient(135deg, #0f2a1b 0%, #1f6f43 100%);
    font-family: system-ui, "Segoe UI", "Noto Sans CJK KR", "Malgun Gothic", sans-serif; display: flex; align-items: center; }
  .txt { padding: 0 80px; width: 740px; position: relative; z-index: 1; }
  h1 { font-size: 104px; line-height: 1; margin: 0; letter-spacing: -3px; }
  .ko { font-size: 36px; opacity: .8; margin: 10px 0 30px; }
  p { font-size: 40px; line-height: 1.25; margin: 0 0 36px; }
  .chips span { display: inline-block; background: rgba(255,255,255,.16); border: 2px solid rgba(255,255,255,.4);
    border-radius: 999px; padding: 8px 22px; font-size: 26px; margin: 0 12px 10px 0; }
  svg { position: absolute; right: -40px; top: 0; opacity: .45; }
</style>
<div class="txt"><h1>FutsalCut</h1><div class="ko">풋살컷</div>
  <p>Keep only the play from your game recordings</p>
  <div class="chips"><span>Free</span><span>No upload</span><span>Original quality</span></div></div>
<svg width="560" height="630" viewBox="0 0 560 630" fill="none" stroke="#fff" stroke-width="6">
  <rect x="40" y="60" width="480" height="510"/><line x1="40" y1="315" x2="520" y2="315"/>
  <circle cx="280" cy="315" r="90"/><rect x="150" y="60" width="260" height="110"/><rect x="150" y="460" width="260" height="110"/>
  <circle cx="280" cy="315" r="8" fill="#fff"/></svg>`;

const browser = await chromium.launch({ channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
await page.setContent(html);
await page.screenshot({ path: out });
await browser.close();
console.log('wrote', out);
