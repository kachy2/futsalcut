/** The app's root folder (where models/ and coi-sw.js live). Both entry pages, / and /ko/, load the same bundle from
 * <root>/assets/, so this is right for either, unlike the page URL (/ko/models/... does not exist). */
export const APP_ROOT = new URL(/* @vite-ignore */ '..', import.meta.url).href;
