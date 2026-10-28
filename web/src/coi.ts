import { APP_ROOT } from './base';

const GUARD = 'coi-reload';

/**
 * True if the app may reload itself: two attempts a minute at most, so a worker that does not isolate the page cannot make a loop.
 * The attempts are kept in localStorage: the isolated page (COOP same-origin) and the page of a forced refresh (no COOP header) are in
 * different browsing context groups, and sessionStorage changes made in one came back stale in the other.
 */
function mayReload(): boolean {
  try {
    const now = Date.now();
    const stored: unknown = JSON.parse(localStorage.getItem(GUARD) ?? '[]');
    const recent = (Array.isArray(stored) ? (stored as number[]) : []).filter((t) => now - t < 60_000);
    if (recent.length >= 2) return false;
    localStorage.setItem(GUARD, JSON.stringify([...recent, now]));
  } catch { /* storage blocked: go on, the worker cannot be registered without storage either */ }
  return true;
}

/**
 * GitHub Pages cannot send the COOP/COEP headers that SharedArrayBuffer needs. A tiny service worker (public/coi-sw.js)
 * adds them, which lets onnxruntime-web run the person detector on several threads. The first visit reloads once;
 * without service workers (private windows, old browsers) the page still works, on one thread.
 */
export function ensureIsolation(): void {
  if (import.meta.env.DEV || window.crossOriginIsolated || !window.isSecureContext || !('serviceWorker' in navigator)) return;
  const reload = () => { if (mayReload()) location.reload(); };
  navigator.serviceWorker.register(new URL('coi-sw.js', APP_ROOT)).then((reg) => {
    // the first activation takes control of this page (controllerchange); a forced refresh (Ctrl+F5) bypasses an active worker
    if (reg.active && !navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener('controllerchange', reload, { once: true });
  }).catch(() => undefined);
}

/** Forget the reload attempts once the page is isolated, so that a later forced refresh can repair itself. */
export function clearIsolationGuard(): void {
  if (!window.crossOriginIsolated) return;
  try { localStorage.removeItem(GUARD); } catch { /* ignore */ }
}
