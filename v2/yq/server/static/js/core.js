// Core helpers shared by every screen: DOM builder, REST calls, client event hub, toasts.

export const $ = (sel, root = document) => root.querySelector(sel);

// h("div", {class: "x", onclick: fn}, "text", child, [more]) -> Element
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  add(el, children);
  return el;
}

function add(el, children) {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    if (Array.isArray(c)) add(el, c);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export const svg = (markup, cls = "ico") => h("span", { class: cls, html: markup, "aria-hidden": "true" });

// --- client event hub (server events are re-emitted here by type) -------------------------
const listeners = new Map();
export const hub = {
  on(type, fn) {
    if (!listeners.has(type)) listeners.set(type, new Set());
    listeners.get(type).add(fn);
    return () => listeners.get(type)?.delete(fn);
  },
  emit(type, msg) {
    for (const fn of [...(listeners.get(type) || [])]) {
      try { fn(msg); } catch (e) { console.error("handler for", type, e); }
    }
  },
};

// --- REST -------------------------------------------------------------------------------------
async function post(url, body) {
  try {
    const r = await fetch(url, { method: "POST", headers: { "content-type": "application/json" },
                                 body: JSON.stringify(body || {}) });
    let data = {};
    try { data = await r.json(); } catch (_) { /* empty body */ }
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    toast("sin conexión con el robot", "error");
    return { ok: false, status: 0, data: {} };
  }
}

export const api = {
  post,
  start: (flow, extra) => post("/api/start", Object.assign({ flow }, extra || {})),
  act: (action, payload) => post("/api/act/" + encodeURIComponent(action), payload || {}),
  cancel: () => post("/api/cancel"),
  async get(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(url + " -> " + r.status);
    return r.json();
  },
};

// --- toasts --------------------------------------------------------------------------------------
let toastTimer = null;
export function toast(text, kind = "info", ms = 3200) {
  let el = $("#toast");
  if (!el) { el = h("div", { id: "toast", role: "status" }); document.body.append(el); }
  el.className = "toast" + (kind === "error" ? " error" : "");
  el.textContent = text;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), ms);
}

// fire a button's action once, with pressed feedback, ignoring double taps
export function tap(fn, ms = 450) {
  let busy = false;
  return (ev) => {
    if (busy) return;
    busy = true;
    const b = ev && ev.currentTarget;
    if (b && b.classList) { b.classList.add("pressed"); setTimeout(() => b.classList.remove("pressed"), 140); }
    try { fn(ev); } finally { setTimeout(() => { busy = false; }, ms); }
  };
}

export const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
export const fold = (s) => String(s || "").normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase();
export const pct = (x) => Math.round(Math.max(0, Math.min(1, Number(x) || 0)) * 100);
