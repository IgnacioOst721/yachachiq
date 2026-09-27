// Story, part 2: confirm the text, consent photo, drawing progress, show time, QR goodbye.
import { h, svg, api, tap, esc, pct } from "../core.js";
import { icons, chakana } from "../icons.js";
import { t } from "../i18n.js";
import { openKeyboard, closeKeyboard } from "../keyboard.js";
import { langPicker } from "../langpicker.js";
import { qrSvg } from "../qr.js";

const STAGES = ["planning", "generating", "verifying", "vectorizing", "layout"];
const PRINTER = ["queued", "drawing_front", "flipping", "drawing_back", "done"];
const printerText = (p) => t("printer_" + ((p && p.status) || "none"));

function langPickOverlay() {
  const ov = h("div", { class: "overlay" }, h("div", { class: "sheet" }, h("h2", {}, t("confirm_other_lang")),
    langPicker({ url: "/api/languages?feature=story", auto: false, onPick: (code) => { ov.remove(); api.act("relang", { code }); } }),
    h("div", { class: "bar-actions" }, h("button", { class: "btn", onclick: () => { closeKeyboard(); ov.remove(); } }, t("kb_close")))));
  document.body.append(ov);
}

let refs = {};

export const screens = {
  confirm: {
    render(d) {
      const text = h("div", { class: "story-text" }, d.text);
      const es = h("div", { class: "story-es" });
      refs = { text, es };
      const cands = (d.candidates || []).filter((c) => c[0] !== d.lang).slice(0, 3);
      const showEs = d.lang !== "spa_Latn";
      screens.confirm.update(d);
      return h("section", { class: "confirm" },
        h("div", { class: "s-head" }, h("div", {}, h("div", { class: "kicker" }, t("confirm_kicker")), h("h2", {}, t("confirm_title"))),
          h("span", { class: "badge big-badge" }, svg(d.source === "sign" ? icons.hand : d.source === "text" ? icons.keyboard : icons.mic), d.lang_name)),
        h("div", { class: "confirm-grid" + (showEs ? " two" : "") },
          h("div", { class: "panel story-panel" }, h("div", { class: "label" }, t("confirm_yours")), text),
          showEs ? h("div", { class: "panel dark story-panel" }, h("div", { class: "label" }, t("confirm_es")), es) : null),
        d.source === "voice" && d.candidates && d.candidates.length ? h("div", { class: "row relang" },
          h("span", {}, t("confirm_detected", { lang: d.lang_name })),
          cands.map((c) => h("button", { class: "chip", onclick: tap(() => api.act("relang", { code: c[0] })) }, c[2] || c[0])),
          h("button", { class: "chip", onclick: tap(langPickOverlay) }, t("confirm_other"))) : null,
        h("div", { class: "bar-actions" },
          h("button", { class: "btn ghost left", onclick: tap(() => api.act("retell")) }, svg(icons.retry), t("confirm_retell")),
          h("button", { class: "btn turq", onclick: tap(() => openKeyboard({ text: refs.text.textContent, title: t("confirm_fix"),
            doneLabel: t("save"), onDone: (v) => v.trim() && api.act("edit_text", { text: v }) })) }, svg(icons.pencil), t("confirm_fix")),
          h("button", { class: "btn big green", onclick: tap(() => api.act("confirm")) }, svg(icons.check), t("confirm_yes"))));
    },
    update(d) {
      if (!refs.text) return;
      refs.text.textContent = d.text;
      if (d.translating) refs.es.replaceChildren(h("span", { class: "spin-inline", html: chakana }), " " + t("translating"));
      else if (d.text_es) refs.es.textContent = d.text_es;
      else refs.es.replaceChildren(h("span", { class: "muted" }, t("translate_failed")));
    },
    leave() { closeKeyboard(); refs = {}; },
  },

  consent: {
    render(d, ctx) {
      const count = h("div", { class: "count" }, d.seconds);
      const st = h("div", { class: "cam-state" }, t("consent_looking"));
      const img = h("img", { class: "cam-img", src: d.preview + "?t=" + Date.now(), alt: "" });
      ctx.on("consent_tick", (m) => {
        count.textContent = m.remaining;
        const [cls, key] = !m.camera ? ["", "consent_nocam"] : m.covered ? ["no", "consent_covered"]
          : m.smile ? ["ok", "consent_smile"] : m.face ? ["ok", "consent_face"] : ["", "consent_open"];
        st.className = "cam-state " + cls;
        st.textContent = t(key);
      });
      return h("section", { class: "consent" },
        h("div", { class: "consent-text" },
          h("div", { class: "kicker" }, t("consent_kicker")), h("h2", {}, t("consent_title")),
          h("p", { class: "lead" }, t("consent_lead")),
          h("div", { class: "rule" }, svg(icons.hand), h("div", {}, h("b", {}, t("consent_no_t")), h("span", {}, t("consent_no_d")))),
          h("div", { class: "rule" }, svg(icons.camera), h("div", {}, h("b", {}, t("consent_yes_t")), h("span", {}, t("consent_yes_d")))),
          h("div", { class: "row" },
            h("button", { class: "btn big green", onclick: tap(() => api.act("consent", { publish: true })) }, svg(icons.check), t("consent_yes")),
            h("button", { class: "btn big", onclick: tap(() => api.act("consent", { publish: false })) }, svg(icons.x), t("consent_no"))),
          h("p", { class: "fine" }, t("consent_fine"))),
        h("div", { class: "cam-box consent-cam" }, img, count, st));
    },
    leave() { const img = document.querySelector(".consent .cam-img"); if (img) img.removeAttribute("src"); },
  },

  consent_result: {
    render(d) {
      return h("section", { class: "center big-msg " + (d.publish ? "yes" : "no") },
        svg(d.publish ? icons.check : icons.hand, "ico huge"),
        h("h1", {}, t(d.publish ? "consent_thanks" : "consent_private")),
        h("p", { class: "lead" }, t(d.publish ? "consent_thanks_d" : "consent_private_d")));
    },
  },

  making: {
    render(d) {
      const steps = h("ol", { class: "steps" }, STAGES.map((s) => h("li", { "data-s": s }, t("stage_" + s))));
      const bar = h("i", {});
      const msg = h("p", { class: "lead msg" });
      const art = h("div", { class: "paper" }, h("div", { class: "pencil", html: icons.pencil }));
      refs = { steps, bar, msg, art, img: null };
      screens.making.update(d);
      return h("section", { class: "making" },
        h("div", { class: "making-side" }, h("div", { class: "kicker" }, t("making_kicker")), h("h2", {}, t("making_title")),
          steps, h("div", { class: "weave" }, bar), msg),
        art);
    },
    update(d) {
      if (!refs.steps) return;
      const alias = { plan: "planning", offline: "generating", image: "generating", verify: "verifying", tracing: "vectorizing" };
      const i = d.stage === "done" ? STAGES.length : STAGES.indexOf(alias[d.stage] || d.stage);
      refs.steps.querySelectorAll("li").forEach((li, k) => { li.className = k < i ? "done" : k === i ? "now" : ""; });
      refs.bar.style.width = pct(d.fraction) + "%";
      refs.msg.textContent = d.message_es || "";
      if (d.image && refs.img !== d.image) {
        refs.img = d.image;
        refs.art.replaceChildren(h("img", { src: d.image, alt: "", class: "reveal" }));
      }
    },
    leave() { refs = {}; },
  },

  showtime: {
    render(d, ctx) {
      const sents = (d.narration && d.narration.sentences) || [];
      const story = h("div", { class: "narr" }, sents.map((s, i) => h("span", { "data-i": i }, s + " ")));
      const pr = h("div", { class: "stat-card" }, svg(icons.printer), h("div", {}, h("b", {}, t("printer")), h("span", { class: "v" })));
      const ho = h("div", { class: "stat-card" }, svg(icons.holo), h("div", {}, h("b", {}, t("hologram")), h("span", { class: "v" })));
      refs = { story, pr, ho };
      ctx.on("printer", (m) => screens.showtime.update(Object.assign(ctx.data, { printer: m })));
      screens.showtime.update(d);
      return h("section", { class: "showtime" },
        h("div", { class: "paper big" }, h("img", { src: d.image, alt: "" })),
        h("div", { class: "show-side" }, h("div", { class: "kicker" }, t("show_kicker")), h("h2", {}, d.title), story,
          h("div", { class: "row stats" }, pr, ho),
          h("div", { class: "bar-actions" }, h("button", { class: "btn ghost", onclick: tap(() => api.act("skip_narration")) }, svg(icons.skip), t("skip")))));
    },
    update(d) {
      if (!refs.story) return;
      const idx = d.narration ? d.narration.index : -1;
      refs.story.querySelectorAll("span").forEach((s) => { const i = +s.dataset.i; s.className = i < idx ? "said" : i === idx ? "now" : ""; });
      const p = d.printer || {};
      refs.pr.querySelector(".v").textContent = printerText(p) + (p.progress != null && p.status !== "done" ? " · " + pct(p.progress) + "%" : "");
      refs.pr.dataset.s = p.status || "";
      const ho = d.hologram || {};
      refs.ho.querySelector(".v").textContent = t("holo_" + (ho.status || "none"));
      refs.ho.dataset.s = ho.status || "";
    },
    leave() { refs = {}; },
  },

  done: {
    render(d, ctx) {
      const pr = h("span", { class: "v" }, printerText(d.printer));
      ctx.on("printer", (m) => { pr.textContent = printerText(m) + (m.progress != null && m.status !== "done" ? " · " + pct(m.progress) + "%" : ""); });
      return h("section", { class: "done" },
        h("div", { class: "qr-card" }, h("div", { class: "qr", html: qrSvg(d.qr_url) }),
          h("b", {}, t(d.public ? "done_qr_public" : "done_qr_private"))),
        h("div", { class: "done-side" },
          h("div", { class: "kicker" }, t("done_kicker")), h("h1", {}, t("done_title")), h("h3", {}, d.title),
          h("div", { class: "row" }, h("img", { class: "thumb", src: d.image, alt: "" }),
            h("div", { class: "stat-card" }, svg(icons.printer), h("div", {}, h("b", {}, t("printer")), pr))),
          h("p", { class: "lead" }, t(d.public ? "done_public" : "done_private")),
          h("div", { class: "bar-actions" },
            h("button", { class: "btn ghost", onclick: tap(() => api.act("finish")) }, svg(icons.home), t("finish")),
            h("button", { class: "btn big magenta", onclick: tap(() => api.act("new_story")) }, svg(icons.mic), t("new_story")))));
    },
  },
};
