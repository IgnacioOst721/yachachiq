// Interface texts: static/i18n/<lang>.json. Spanish is the base; missing keys fall back to Spanish.
// Messages that come from the server (message_es) are Spanish-only unless they carry a "key".
import { hub } from "./core.js";

const cache = {};
let lang = "es";

async function load(l) {
  if (!cache[l]) {
    const r = await fetch(`/static/i18n/${l}.json`, { cache: "no-cache" });
    cache[l] = r.ok ? await r.json() : {};
  }
  return cache[l];
}

export async function initI18n() {
  await load("es");
}

export function t(key, vars) {
  let s = (cache[lang] && cache[lang][key]) ?? (cache.es && cache.es[key]) ?? key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split("{" + k + "}").join(String(v));
  return s;
}

// server message: use the translated key when there is one, else the Spanish text
export const tm = (key, message_es) => (key && (cache[lang]?.[key] || cache.es?.[key])) ? t(key) : (message_es || "");

export function getLang() { return lang; }

export async function setLang(l) {
  if (!["es", "en", "qu"].includes(l)) l = "es";
  await load(l);
  lang = l;
  document.documentElement.lang = l === "qu" ? "qu" : l;
  hub.emit("lang", l);
}
