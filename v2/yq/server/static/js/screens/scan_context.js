// Scan, optional step "¿Dónde lo encontraron?": region chips + free text (keyboard or dictation).
// Sent as ScanRequest.context: a clue for identification, never proof (CONTRACTS.md §9).
import { h, svg, api, tap, esc } from "../core.js";
import { icons } from "../icons.js";
import { t, getLang } from "../i18n.js";
import { openKeyboard, closeKeyboard } from "../keyboard.js";

let st = {};

export const regionLabel = (code) => (code ? t("region_" + code) : "");

function drawText() {
  const { box, text } = st;
  if (!box) return;
  box.innerHTML = text ? esc(text) : '<span class="muted">' + esc(t("ctx_placeholder")) + "</span>";
  st.clear.classList.toggle("hidden", !text);
}

function drawMic(d) {
  const b = st.mic;
  if (!b) return;
  b.classList.toggle("recording", !!d.recording);
  b.disabled = !!d.transcribing;
  b.lastChild.textContent = d.recording ? t("ctx_stop") : d.transcribing ? t("ctx_transcribing") : t("ctx_dictate");
}

export const screens = {
  context: {
    render(d, ctx) {
      st = { text: d.found_where || "", region: d.region_hint || "", typed: false };
      const chips = h("div", { class: "region-grid" }, (d.regions || []).map((r) =>
        h("button", { class: "chip region" + (r === st.region ? " on" : "") + (r === "no_se" ? " unknown" : ""), "data-r": r,
          onclick: tap(() => {
            st.region = st.region === r ? "" : r;
            chips.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.dataset.r === st.region));
            api.act("set_context", { region_hint: st.region });
          }, 200) }, regionLabel(r))));
      st.box = h("div", { class: "panel ctx-text", onclick: tap(() => write()) });
      st.clear = h("button", { class: "btn", onclick: tap(() => { st.text = ""; st.typed = true; drawText();
        api.act("set_context", { found_where: "", lang: getLang() }); }) }, svg(icons.trash), t("sign_clear"));
      const write = () => openKeyboard({ text: st.text, title: t("ctx_where"), doneLabel: t("save"), max: d.max_chars || 200,
        placeholder: t("ctx_placeholder"),
        onChange: (v) => { st.text = v; drawText(); },
        onDone: (v) => { st.text = v; st.typed = true; drawText(); api.act("set_context", { found_where: v, lang: getLang() }); } });
      st.mic = h("button", { class: "btn magenta", onclick: tap(() => {
        if (st.mic.classList.contains("recording")) api.act("stop_dictation");
        else { closeKeyboard(); api.act("dictate", { ui_lang: getLang() }); }
      }) }, svg(icons.mic), h("span", {}, t("ctx_dictate")));
      ctx.on("level", (m) => st.mic && st.mic.style.setProperty("--lvl", Math.min(1, m.level * 1.4).toFixed(2)));
      drawText();
      drawMic(d);
      const cont = () => {
        const body = { region_hint: st.region, found_where: st.text };
        if (st.typed) body.lang = getLang();
        api.act("continue", body);
      };
      return h("section", { class: "scan-context" },
        h("div", { class: "s-head" }, h("div", {}, h("div", { class: "kicker" }, t("scan_kicker")), h("h2", {}, t("ctx_title"))),
          h("span", { class: "badge big-badge" }, t("ctx_optional"))),
        h("p", { class: "lead" }, t("ctx_lead")),
        h("div", { class: "label" }, t("ctx_region")), chips,
        h("div", { class: "label" }, t("ctx_where")),
        h("div", { class: "ctx-row" }, st.box, h("div", { class: "ctx-btns" },
          h("button", { class: "btn turq", onclick: tap(() => write()) }, svg(icons.keyboard), t("ctx_type")), st.mic, st.clear)),
        h("div", { class: "bar-actions" },
          h("button", { class: "btn ghost left", onclick: tap(() => api.act("back")) }, svg(icons.back), t("back")),
          h("button", { class: "btn big ghost", onclick: tap(() => { closeKeyboard(); api.act("skip"); }) }, svg(icons.skip), t("ctx_skip")),
          h("button", { class: "btn big turq", onclick: tap(() => { closeKeyboard(); cont(); }) }, svg(icons.play), t("ctx_continue"))));
    },
    update(d, ctx, partial) {
      if ("found_where" in (partial || {}) && partial.dictated) { st.text = d.found_where || ""; st.typed = false; drawText(); }
      if ("region_hint" in (partial || {})) st.region = d.region_hint || "";
      drawMic(d);
    },
    leave() { closeKeyboard(); st = {}; },
  },
};
