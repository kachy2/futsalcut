/* Cross-origin isolation for hosts that cannot set response headers (GitHub Pages).
 * SharedArrayBuffer, and with it multi-threaded WebAssembly for the person detector, needs these two headers on
 * the page. This worker adds them to the app's own responses; it caches nothing and touches no other site.
 * Pattern from coi-serviceworker (MIT, Guido Zuidhof). Registered by src/coi.ts. */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', (e) => {
  const r = e.request;
  if (r.cache === 'only-if-cached' && r.mode !== 'same-origin') return;
  e.respondWith(fetch(r).then((res) => {
    if (res.status === 0) return res; // opaque (cross-origin or redirect): cannot be rewritten
    const headers = new Headers(res.headers);
    headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
    headers.set('Cross-Origin-Opener-Policy', 'same-origin');
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  }).catch(() => Response.error()));
});
