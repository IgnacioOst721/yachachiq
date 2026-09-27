// Yachachiq kiosk SPA: the server says which screen to show ("screen" events); this file routes
// them to screen modules, keeps the WebSocket alive and runs the header, idle and language UI.
import { $, h, hub, api, tap, toast } from "./core.js";
import { initI18n, setLang, getLang, t } from "./i18n.js";
import { chakana, icons } from "./icons.js";
import { openGear } from "./gear.js";
import { screens as home } from "./screens/home.js";
import { screens as storyIn } from "./screens/story_input.js";
import { screens as storyOut } from "./screens/story_output.js";
import { screens as scan } from "./screens/scan.js";
import { screens as results } from "./screens/results.js";
import { screens as common } from "./screens/common.js";

const SCREENS = Object.assign({}, home, storyIn, storyOut, scan, results, common);
const cur = { key: "", name: "", flow: null, def: null, data: {}, offs: [] };
let ws = null, bootId = null, lastActivity = 0;

// --- router ------------------------------------------------------------------------------------
function ctx() {
  return {
    on: (type, fn) => { cur.offs.push(hub.on(type, fn)); },
    get data() { return cur.data; },
    rerender: () => render(cur.flow, cur.name, cur.data, true),
  };
}

function leave() {
  cur.offs.forEach((off) => off());
  cur.offs = [];
  try { cur.def && cur.def.leave && cur.def.leave(); } catch (e) { console.error(e); }
}

function render(flow, name, data, force = false) {
  const key = (flow || "") + ":" + name;
  const def = SCREENS[name] || SCREENS.unknown;
  if (!force && key === cur.key && def.update) {
    cur.data = data || {};
    def.update(cur.data, ctx(), cur.data);
    return;
  }
  leave();
  Object.assign(cur, { key, name, flow, def, data: data || {} });
  const stage = $("#stage");
  let el;
  try { el = def.render(cur.data, ctx()); } catch (e) { console.error("render", name, e); el = SCREENS.unknown.render({ name }); }
  el.classList.add("screen", "s-" + name);
  [...el.children].forEach((c, i) => c.style.setProperty("--i", i));
  stage.replaceChildren(el);
  document.body.dataset.flow = flow || "home";
  document.body.dataset.screen = name;
  $("#homeBtn").classList.toggle("hidden", !flow);
  resetHomeBtn();
}

function patch(partial) {
  Object.assign(cur.data, partial || {});
  if (cur.def && cur.def.update) cur.def.update(cur.data, ctx(), partial || {});
  else render(cur.flow, cur.name, cur.data, true);
}

// --- server events -------------------------------------------------------------------------------
function onMessage(msg) {
  switch (msg.type) {
    case "hello":
      if (bootId && msg.boot_id !== bootId) { location.reload(); return; }
      bootId = msg.boot_id;
      render(msg.flow, msg.screen, msg.data, true);
      break;
    case "screen":
      if (msg.screen === "home" && msg.data && msg.data.reset && getLang() !== "es") setLang("es");
      hideIdle();
      render(msg.flow, msg.screen, msg.data);
      break;
    case "screen_update": if (cur.name === msg.screen) patch(msg.data); break;
    case "idle_warning": showIdle(msg.seconds); break;
    case "idle_clear": hideIdle(); break;
    case "toast": toast(msg.key ? t(msg.key) : msg.text_es, msg.kind); break;
    case "status": setStatus(msg); break;
  }
  hub.emit(msg.type, msg);
}

function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => document.body.classList.remove("offline");
  ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  ws.onclose = () => { document.body.classList.add("offline"); setTimeout(connect, 1500); };
}

// every touch keeps the visitor's screen alive (the server runs the idle timer)
function activity() {
  const now = Date.now();
  if (now - lastActivity < 700) return;
  lastActivity = now;
  if (ws && ws.readyState === 1) ws.send('{"type":"activity"}');
  hideIdle();
}

// --- idle overlay --------------------------------------------------------------------------------------
let idleTimer = null;
function showIdle(seconds) {
  hideIdle();
  let left = Math.max(1, Math.round(seconds || 10));
  const count = h("div", { class: "count" }, left);
  const box = h("div", { class: "overlay", id: "idle", onpointerdown: activity },
    h("div", { class: "sheet idle-box" }, h("h2", {}, t("idle_title")), count, h("p", { class: "lead" }, t("idle_body"))));
  document.body.append(box);
  idleTimer = setInterval(() => { left = Math.max(0, left - 1); count.textContent = left; }, 1000);
}
function hideIdle() { clearInterval(idleTimer); const el = $("#idle"); if (el) el.remove(); }

// --- header --------------------------------------------------------------------------------------------
let homeArmed = null;
function resetHomeBtn() {
  clearTimeout(homeArmed); homeArmed = null;
  const b = $("#homeBtn");
  b.classList.remove("armed");
  b.lastChild.textContent = t("home");
}
function onHome() {
  const b = $("#homeBtn");
  if (!homeArmed) {                  // two taps: nobody loses a story by accident
    b.classList.add("armed");
    b.lastChild.textContent = t("home_confirm");
    homeArmed = setTimeout(resetHomeBtn, 4000);
    return;
  }
  resetHomeBtn();
  api.cancel();
}

function setStatus(st) {
  const r = st.reachable || {};
  for (const k of ["mac", "printer", "hologram"]) {
    const dot = $("#dot-" + k);
    if (!dot) continue;
    const mode = (st.subsystems || {})[k] || "";
    dot.className = r[k] ? (String(mode).startsWith("mock") ? "mock" : "on") : "";
  }
}

function header() {
  const langBtn = (code) => h("button", { "aria-pressed": String(getLang() === code), "data-lang": code,
    onclick: tap(() => setLang(code)) }, code.toUpperCase());
  const top = h("header", { class: "top" },
    h("div", { class: "brand" }, h("span", { html: chakana, "aria-hidden": "true" }),
      h("div", {}, h("b", {}, "Yachachiq"), h("small", { id: "tagline" }, t("tagline")))),
    h("div", { class: "spacer" }),
    h("button", { id: "homeBtn", class: "home-btn hidden", onclick: onHome }, h("span", { html: icons.home }), h("span", {}, t("home"))),
    h("div", { class: "dots", title: "Mac · impresora · holograma" },
      h("i", { id: "dot-mac" }), "Mac", h("i", { id: "dot-printer" }), t("dot_printer"), h("i", { id: "dot-hologram" }), t("dot_holo")),
    h("div", { class: "langs", role: "group", "aria-label": "idioma" }, langBtn("es"), langBtn("en"), langBtn("qu")),
    h("button", { class: "gear", "aria-label": "ajustes", html: icons.gear, onclick: tap(openGear) }));
  document.body.prepend(top);
}

async function boot() {
  await initI18n();
  header();
  hub.on("lang", (l) => {
    document.querySelectorAll(".langs button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === l)));
    $("#tagline").textContent = t("tagline");
    resetHomeBtn();
    if (cur.name) render(cur.flow, cur.name, cur.data, true);
  });
  document.addEventListener("pointerdown", activity, { capture: true, passive: true });
  document.addEventListener("contextmenu", (e) => e.preventDefault());   // long-press menus off
  if (new URLSearchParams(location.search).has("debug")) {       // preview any screen: yqRender("story","listening",{})
    window.yqRender = (flow, name, data) => render(flow, name, data || {}, true);
    window.yqEmit = (msg) => onMessage(msg);
  }
  connect();
  fetch("/status").then((r) => r.json()).then(setStatus).catch(() => {});
}

boot();
