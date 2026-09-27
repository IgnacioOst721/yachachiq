// Searchable language list: "Detectar automáticamente" first, Peruvian languages, then the rest.
import { h, svg, api, tap, fold } from "./core.js";
import { icons } from "./icons.js";
import { t, getLang } from "./i18n.js";
import { openKeyboard, closeKeyboard } from "./keyboard.js";

const cache = {};

export async function languages(url) {
  if (!cache[url]) cache[url] = api.get(url).then((d) => d.languages || []).catch(() => { delete cache[url]; return []; });
  return cache[url];
}

export function langLabel(l) {
  if (!l) return "";
  return getLang() === "en" && l.en ? l.en : l.name;
}

// returns an element; onPick(code, lang) is called once per tap
export function langPicker({ url, selected, auto = true, onPick }) {
  let query = "";
  const list = h("div", { class: "lang-list", role: "list" }, h("p", { class: "muted" }, t("loading")));
  const qBox = h("span", { class: "q" }, t("lang_search"));
  const search = h("button", { class: "chip search", onclick: tap(() => openKeyboard({
    mode: "search", text: query, title: t("lang_search"), doneLabel: t("kb_done"),
    suggest: (v) => { const q = fold(v); return q ? all.filter((l) => [l.name, l.native, l.en, l.code].some((x) => fold(x).includes(q)))
      .slice(0, 5).map(chip) : [h("span", { class: "muted" }, t("lang_type_hint"))]; },
    onChange: (v) => { query = v; qBox.textContent = v || t("lang_search"); draw(); },
    onDone: (v) => { query = v; qBox.textContent = v || t("lang_search"); draw(); } })) }, svg(icons.search), qBox);
  let all = [];

  const chip = (l) => h("button", { class: "chip lang" + (l.code === selected ? " on" : ""), role: "listitem",
    onclick: tap(() => { closeKeyboard(); onPick(l.code, l); }) },
    h("span", {}, langLabel(l)), l.native && l.native !== l.name ? h("small", {}, l.native) : null);

  function draw() {
    const q = fold(query);
    const hit = (l) => !q || [l.name, l.native, l.en, l.code].some((s) => fold(s).includes(q));
    const peru = all.filter((l) => l.peru && hit(l));
    const rest = all.filter((l) => !l.peru && hit(l));
    const blocks = [];
    if (auto && !q) {
      blocks.push(h("button", { class: "chip lang auto" + (selected === "auto" ? " on" : ""),
        onclick: tap(() => { closeKeyboard(); onPick("auto", null); }) }, svg(icons.sparkle), t("lang_auto")));
    }
    if (peru.length) blocks.push(h("div", { class: "lang-sec" }, t("lang_peru")), h("div", { class: "lang-grid" }, peru.map(chip)));
    if (rest.length) blocks.push(h("div", { class: "lang-sec" }, t("lang_other")), h("div", { class: "lang-grid" }, rest.map(chip)));
    if (!peru.length && !rest.length) blocks.push(h("p", { class: "lead muted" }, t("lang_none")));
    list.replaceChildren(...blocks);
  }

  languages(url).then((ls) => { all = ls; draw(); });
  return h("div", { class: "picker" }, h("div", { class: "row" }, search), list);
}
