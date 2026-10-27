// Shared steps of the end-to-end tests: a Chrome with a persistent profile, the app opened and isolated, a drawn zone.
import { chromium, expect, type BrowserContext, type Page } from '@playwright/test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const VIDEO = process.env.APP_VIDEO ?? '/data/synthetic/syn_scene.mp4';
export type Zone = [number, number][];

/** Persistent profile: the default (incognito-like) context has a tiny storage quota for the OPFS stand-in of the save dialog. */
export async function launch(baseURL: string | undefined): Promise<{ ctx: BrowserContext; page: Page; logs: string[] }> {
  const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), 'futsalcut-e2e-')), { channel: 'chrome', headless: true, baseURL,
    args: (process.env.CHROME_ARGS ?? '').split(' ').filter(Boolean) }); // e.g. --enable-unsafe-webgpu for a software WebGPU adapter
  const page = ctx.pages()[0] ?? (await ctx.newPage());
  const logs: string[] = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.addInitScript(() => {
    // headless runs cannot click through the native save dialog: write into the origin private file system
    (window as unknown as { showSaveFilePicker: unknown }).showSaveFilePicker = async () => {
      const root = await navigator.storage.getDirectory();
      return root.getFileHandle('e2e_output.mp4', { create: true });
    };
  });
  return { ctx, page, logs };
}

/** Open the app (ORT_THREADS=n adds ?threads=n; APP_QUERY adds any query, e.g. ?gpuwait=600), wait until the service worker has
 * isolated the page, pick the video, go to the zone step. */
export async function openApp(page: Page): Promise<void> {
  const query = process.env.APP_QUERY ?? (process.env.ORT_THREADS ? `?threads=${process.env.ORT_THREADS}` : '');
  await page.goto((process.env.APP_URL ? '' : '/') + query); // '' = the base URL itself (keeps a /<repo>/ path)
  // the first visit installs the service worker and reloads once; WASM threads need the isolated page
  await expect.poll(() => page.evaluate(() => crossOriginIsolated).catch(() => false), { timeout: 60_000 }).toBe(true);
  await page.setInputFiles('input[type=file]', VIDEO);
  await page.getByRole('button', { name: /다음|Next/ }).click();
  await expect(page.locator('canvas')).toBeVisible({ timeout: 120_000 });
}

/** Draw the polygon like a user: reset, then click each corner on the preview canvas (scrolled into view first). */
export async function drawZone(page: Page, pts: Zone): Promise<void> {
  await page.getByRole('button', { name: /초기화|Reset/ }).click();
  const canvas = page.locator('canvas');
  const box = (await canvas.boundingBox())!;
  for (const [x, y] of pts) {
    await canvas.click({ position: { x: Math.min(0.999, Math.max(0.001, x)) * box.width, y: Math.min(0.999, Math.max(0.001, y)) * box.height } });
  }
}

export const REVIEW = /결과 확인|Review/;
export const DRAW = /구역 그리기|Draw the zone/;

interface Hook { intervals: [number, number][]; prob: number[]; light: number[]; detector: [number, number][] }

/** Press Analyze and wait for the review step; returns the diagnostics and the result the app exposes for tests. */
export async function analyze(page: Page): Promise<{ diag: Record<string, any>; hook: Hook }> { // eslint-disable-line @typescript-eslint/no-explicit-any
  await page.getByRole('button', { name: /분석 시작|Analyze/ }).click();
  await expect(page.getByRole('heading', { name: REVIEW })).toBeVisible({ timeout: 15 * 60_000 });
  const diag = JSON.parse((await page.locator('details pre').first().textContent()) ?? '{}');
  const hook = await page.evaluate(() => (window as unknown as { __futsalcut: Hook }).__futsalcut);
  return { diag, hook };
}
