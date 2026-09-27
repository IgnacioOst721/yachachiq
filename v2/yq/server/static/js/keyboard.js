// On-screen keyboard, Spanish layout with ñ, accents and the apostrophe Quechua needs (ch', q', t').
// openKeyboard({text, title, mode: "text"|"password"|"search", doneLabel, onChange, onDone, onClose})
import { h, tap } from "./core.js";
import { icons } from "./icons.js";
import { t } from "./i18n.js";

const ROWS = [
  ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "'", "⌫"],
  ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "á", "é"],
  ["a", "s", "d", "f", "g", "h", "j", "k", "l", "ñ", "í", "ó"],
  ["⇧", "z", "x", "c", "v", "b", "n", "m", "ú", "ü", "¿", "?"],
];

let open = null;

export function closeKeyboard() {
  if (open) { open.el.remove(); open.onClose && open.onClose(); open = null; }
}

export function openKeyboard(opts = {}) {
  closeKeyboard();
  const mode = opts.mode || "text";
  let text = opts.text || "";
  let shift = mode === "text" && !text;
  const field = h("div", { class: "field", "aria-live": "polite" });
  const keys = [];

  const draw = () => {
    const shown = mode === "password" ? "•".repeat(text.length) : text;
    field.replaceChildren(shown || "", h("span", { class: "caret" }));
    if (!shown && opts.placeholder) field.prepend(h("span", { class: "muted" }, opts.placeholder + " "));
    field.scrollTop = field.scrollHeight;
    for (const b of keys) {
      if (b.dataset.k.length === 1 && /\p{L}/u.test(b.dataset.k)) b.textContent = shift ? b.dataset.k.toUpperCase() : b.dataset.k;
      if (b.dataset.k === "⇧") b.classList.toggle("on", shift);
    }
  };
  const sugg = h("div", { class: "ksuggest" });
  const suggest = () => { if (opts.suggest) sugg.replaceChildren(...(opts.suggest(text) || [])); };
  const changed = () => { draw(); opts.onChange && opts.onChange(text); suggest(); };
  const type = (k) => {
    if (k === "⌫") { text = text.slice(0, -1); }
    else if (k === "⇧") { shift = !shift; draw(); return; }
    else {
      const ch = shift && /\p{L}/u.test(k) ? k.toUpperCase() : k;
      text += ch;
      if (shift && /\p{L}/u.test(k)) shift = false;
    }
    if (mode === "text" && (/[.!?]\s$/.test(text) || text === "")) shift = true;
    changed();
  };

  const key = (k, cls = "") => {
    const b = h("button", { class: cls, "data-k": k, onpointerdown: (e) => { e.preventDefault(); type(k); } },
      k === "⌫" ? h("span", { html: icons.backspace, style: { display: "inline-block", width: "2.2rem" } }) : k);
    keys.push(b);
    return b;
  };

  const done = h("button", { class: "go", onclick: tap(() => { const v = text; closeKeyboard(); opts.onDone && opts.onDone(v); }) },
    opts.doneLabel || t("kb_done"));
  const close = h("button", { class: "w3", onclick: tap(() => closeKeyboard()) }, t("kb_close"));
  const rows = ROWS.map((r) => h("div", { class: "krow" }, r.map((k) => key(k, k === "⌫" || k === "⇧" ? "w2" : ""))));
  const last = h("div", { class: "krow" }, close, key("¡"), key("!"), key(","), key(" ", "space"), key("."), key("-"), done);
  const el = h("div", { class: "kbd", role: "dialog", "aria-label": t("kb_title") },
    h("div", { class: "kout" }, opts.title ? h("b", { style: { fontSize: "1.2rem", maxWidth: "14rem" } }, opts.title) : null, field),
    opts.suggest ? sugg : null, rows, last);
  document.body.append(el);
  open = { el, onClose: opts.onClose };
  draw();
  suggest();
  return closeKeyboard;
}
