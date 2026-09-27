// Story, part 1: how the visitor tells it (voice / sign / typing).
import { h, svg, api, tap, esc } from "../core.js";
import { icons } from "../icons.js";
import { t } from "../i18n.js";
import { langPicker } from "../langpicker.js";
import { openKeyboard, closeKeyboard } from "../keyboard.js";
import { waiting, backBtn } from "./common.js";

const title = (kicker, text, right) =>
  h("div", { class: "s-head" }, h("div", {}, h("div", { class: "kicker" }, kicker), h("h2", {}, text)), right || null);

let listenTimer = null;

export const screens = {
  method: {
    render(d) {
      const card = (m, cls, icon, name, desc, off) => h("button", { class: "choice " + cls + (off ? " off" : ""),
        onclick: tap(() => api.act("choose", { method: m })) }, svg(icon), h("div", { class: "t" }, name), h("div", { class: "d" }, desc));
      return h("section", {},
        title(t("story_kicker"), t("method_title")),
        h("div", { class: "choices methods" },
          card("voice", "magenta", icons.mic, t("method_voice"), t("method_voice_d")),
          card("sign", "sun", icons.hand, t("method_sign"), d.sign_ok === false ? t("method_sign_off") : t("method_sign_d"), d.sign_ok === false),
          card("text", "green", icons.keyboard, t("method_text"), t("method_text_d"))),
        h("div", { class: "bar-actions" }, h("button", { class: "btn ghost left", onclick: tap(() => api.cancel()) }, svg(icons.back), t("back"))));
    },
  },

  voice_lang: {
    render(d) {
      return h("section", {},
        title(t("voice_kicker"), t("voice_lang_title")),
        h("div", { class: "grow scroll" }, langPicker({ url: d.languages_url, selected: d.selected,
          onPick: (code) => api.act("choose_lang", { code }) })),
        h("div", { class: "bar-actions" }, backBtn("back")));
    },
    leave() { closeKeyboard(); },
  },

  voice_ready: {
    render(d) {
      return h("section", { class: "center mic-stage" },
        h("button", { class: "chip on lang-now", onclick: tap(() => api.act("change_lang")) },
          svg(icons.globe), d.lang === "auto" ? t("lang_auto") : d.lang_name, h("small", {}, t("change"))),
        h("button", { class: "mic-btn pulse", "aria-label": t("tap_talk"), onclick: tap(() => api.act("record")) },
          svg(icons.mic), h("span", {}, t("tap_talk"))),
        h("p", { class: "lead" }, t("voice_hint")),
        h("div", { class: "bar-actions" }, backBtn("back")));
    },
  },

  listening: {
    render(d, ctx) {
      const ring = h("div", { class: "ring" }, h("div", { class: "ring-core" }, svg(icons.mic)));
      const clock = h("div", { class: "clock" }, "0:00");
      const t0 = Date.now();
      clearInterval(listenTimer);
      listenTimer = setInterval(() => {
        const s = Math.floor((Date.now() - t0) / 1000);
        clock.textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
      }, 250);
      ctx.on("level", (m) => ring.style.setProperty("--lvl", Math.min(1, m.level * 1.4).toFixed(3)));
      return h("section", { class: "center mic-stage" },
        h("h2", {}, t("listening")), ring, clock,
        h("p", { class: "lead" }, t("listening_hint")),
        h("button", { class: "btn big sun", onclick: tap(() => api.act("stop_listening")) }, svg(icons.stop), t("done_talking")));
    },
    leave() { clearInterval(listenTimer); },
  },

  transcribing: { render: (d) => waiting(t("transcribing"), d.lang_name && d.lang_name !== "Automático" ? d.lang_name : "") },

  sign_lang: {
    render(d) {
      const colors = ["sun", "magenta", "turq", "green"];
      const short = { prl: "LSP", ase: "ASL", ils: "IS" };
      return h("section", {},
        title(t("sign_kicker"), t("sign_lang_title")),
        h("div", { class: "choices methods" }, (d.sign_langs || []).map((s, i) =>
          h("button", { class: "choice " + colors[i % colors.length] + (s.ready === false ? " off" : ""),
            onclick: tap(() => s.ready !== false && api.act("choose_sign", { code: s.code })) },
            h("div", { class: "tag" }, short[s.code] || s.code.toUpperCase()), svg(icons.hand),
            h("div", { class: "t" }, s.name_es),
            h("div", { class: "d" }, s.ready === false ? t("sign_no_model") : [s.letters ? t("sign_letters") : null,
              s.words ? t("sign_words", { n: s.words }) : null].filter(Boolean).join(" · "))))),
        h("div", { class: "bar-actions" }, backBtn("back")));
    },
  },

  signing: {
    render(d, ctx) {
      const cams = h("img", { class: "cam-img", src: d.preview + "?t=" + Date.now(), alt: "" });
      const hands = h("span", { class: "badge" }, t("sign_hands_wait"));
      const cands = h("div", { class: "cands" }, h("p", { class: "muted" }, t("sign_waiting")));
      const buf = h("div", { class: "buffer" }, h("span", { class: "caret" }));
      let mode = d.letters ? "letters" : "words";
      const modeBtns = d.letters && d.words ? h("div", { class: "langs mode" },
        ["letters", "words"].map((m) => h("button", { "aria-pressed": String(mode === m), "data-m": m,
          onclick: tap(() => { mode = m; api.act("sign_mode", { mode: m });
            modeBtns.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === m))); }) },
          t(m === "letters" ? "sign_mode_letters" : "sign_mode_words")))) : null;
      const letter = h("div", { class: "cam-letter hidden" }, h("b", {}), h("i", {}));
      let lastKey = "";
      ctx.on("sign", (m) => {
        hands.textContent = m.hands_visible ? t("sign_hands_ok") : t("sign_hands_wait");
        hands.className = "badge " + (m.hands_visible ? "ok" : "");
        const list = m.candidates || [];
        const key = list.map((c) => c.text).join("|");
        if (key !== lastKey) {                       // do not rebuild under a finger that is tapping
          lastKey = key;
          cands.replaceChildren(...(list.length ? list.slice(0, 4).map((c, i) =>
            h("button", { class: "cand" + (i === 0 ? " top" : ""), onclick: tap(() => api.act("sign_accept", { index: i }), 250) },
              h("b", {}, c.text), h("small", {}, Math.round((c.prob || 0) * 100) + "%")))
            : [h("p", { class: "muted" }, t("sign_waiting"))]));
        }
        const text = m.text || "", b = m.buffer || "";
        const done = b && text.endsWith(b) ? text.slice(0, text.length - b.length) : text;
        buf.innerHTML = esc(done) + (b && text.endsWith(b) ? '<u class="spelling">' + esc(b) + "</u>" : "") + '<span class="caret"></span>';
        const L = m.letter;
        letter.classList.toggle("hidden", !(L && L.current));
        if (L && L.current) {
          letter.firstChild.textContent = L.current;
          letter.lastChild.style.width = Math.round((L.progress || 0) * 100) + "%";
        }
      });
      return h("section", { class: "signing" },
        h("div", { class: "cam-box" }, cams, letter, h("div", { class: "cam-tags" }, h("span", { class: "badge" }, d.sign_name), hands)),
        h("div", { class: "sign-side" },
          h("div", { class: "row" }, h("div", { class: "label" }, t("sign_what")), h("div", { class: "grow" }), modeBtns),
          cands,
          h("div", { class: "label" }, t("sign_text")),
          buf,
          h("div", { class: "row" },
            h("button", { class: "btn", onclick: tap(() => api.act("sign_backspace"), 200) }, svg(icons.backspace), t("sign_back")),
            d.can_space ? h("button", { class: "btn", onclick: tap(() => api.act("sign_space"), 200) }, svg(icons.space), t("sign_space")) : null,
            h("button", { class: "btn", onclick: tap(() => api.act("sign_clear")) }, svg(icons.trash), t("sign_clear"))),
          h("div", { class: "bar-actions" }, backBtn("back", t("sign_change")),
            h("button", { class: "btn big green", onclick: tap(() => api.act("sign_done")) }, svg(icons.check), t("ready")))));
    },
    leave() { const img = document.querySelector(".cam-img"); if (img) img.removeAttribute("src"); },
  },

  typing: {
    render(d) {
      let text = d.text || "", lang = d.lang || "spa_Latn";
      const quick = [["spa_Latn", "Español"], ["quy_Latn", "Runasimi"], ["eng_Latn", "English"], ["ayr_Latn", "Aymar aru"], ["por_Latn", "Português"]];
      if (!quick.some((q) => q[0] === lang)) quick.push([lang, d.lang_name || lang]);
      const box = h("div", { class: "panel type-box", onclick: tap(() => kb()) });
      const draw = () => { box.innerHTML = text ? esc(text) + '<span class="caret"></span>' : '<span class="muted">' + esc(t("type_placeholder")) + "</span>"; };
      const send = () => { if (text.trim()) api.act("submit_text", { text, lang }); };
      const kb = () => openKeyboard({ text, title: t("type_title"), doneLabel: t("ready"), placeholder: t("type_placeholder"),
        onChange: (v) => { text = v; draw(); }, onDone: (v) => { text = v; draw(); send(); } });
      const chips = h("div", { class: "row" }, h("span", { class: "label" }, t("type_lang")), quick.map(([c, n]) =>
        h("button", { class: "chip" + (c === lang ? " on" : ""), "data-c": c, onclick: tap((e) => { lang = c;
          chips.querySelectorAll(".chip").forEach((b) => b.classList.toggle("on", b.dataset.c === c)); }) }, n)));
      draw();
      setTimeout(kb, 350);
      return h("section", { class: "typing" }, title(t("text_kicker"), t("type_title")), chips, box,
        h("div", { class: "bar-actions" }, backBtn("back"),
          h("button", { class: "btn big green", onclick: tap(send) }, svg(icons.check), t("ready"))));
    },
    leave() { closeKeyboard(); },
  },
};
