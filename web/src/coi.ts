import { APP_ROOT } from './base';

/**
 * GitHub Pages cannot send the COOP/COEP headers that SharedArrayBuffer needs. A tiny service worker (public/coi-sw.js)
 * adds them, which lets onnxruntime-web run the person detector on several threads. The first visit reloads once;
 * without service workers (private windows, old browsers) the page still works, on one thread.
 */
export function ensureIsolation(): void {
  if (import.meta.env.DEV || window.crossOriginIsolated || !window.isSecureContext || !('serviceWorker' in navigator)) return;
  const reload = () => {
    try {
      if (sessionStorage.getItem('coi-reload')) return; // one attempt per tab: never loop
      sessionStorage.setItem('coi-reload', '1');
    } catch { /* storage blocked: reload at most once through the controller change below */ }
    location.reload();
  };
  navigator.serviceWorker.register(new URL('coi-sw.js', APP_ROOT)).then((reg) => {
    // the first activation takes control of this page (controllerchange); a forced refresh bypasses an active worker
    if (reg.active && !navigator.serviceWorker.controller) reload();
    else navigator.serviceWorker.addEventListener('controllerchange', reload, { once: true });
  }).catch(() => undefined);
}

/** Forget the one-reload guard once the page is isolated, so that a later forced refresh can repair itself. */
export function clearIsolationGuard(): void {
  if (!window.crossOriginIsolated) return;
  try { sessionStorage.removeItem('coi-reload'); } catch { /* ignore */ }
}
