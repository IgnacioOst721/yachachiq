// Object analysis: intro, box checks, live scan (turntable ring of photos, lights, thermal).
import { h, svg, api, tap, pct } from "../core.js";
import { icons } from "../icons.js";
import { t } from "../i18n.js";
import { waiting } from "./common.js";

const STAGES = ["weighing", "photogrammetry", "rti", "uv", "thermal", "analyzing"];
const STAGE_ICON = { weighing: icons.scale, photogrammetry: icons.camera, rti: icons.light, uv: icons.uv,
  thermal: icons.thermo, analyzing: icons.sparkle };

function lightName(l) {
  if (!l) return "";
  if (/^led\d+/.test(l)) return t("light_led", { n: l.slice(3) });
  return t("light_" + l) === "light_" + l ? l : t("light_" + l);
}

let refs = {};

export const screens = {
  intro: {
    render(d) {
      const step = (n, icon, key, cls) => h("div", { class: "step-card " + cls }, h("div", { class: "n" }, n), svg(icon), h("p", {}, t(key)));
      const profs = h("div", { class: "row profiles" }, (d.profiles || []).map((p) =>
        h("button", { class: "chip big-chip" + (p.id === d.selected ? " on" : ""), "data-p": p.id,
          onclick: tap(() => api.act("choose_profile", { profile: p.id }), 200) },
          h("span", {}, t("profile_" + p.id)), h("small", {}, t("minutes", { n: p.minutes })))));
      refs = { profs };
      return h("section", { class: "scan-intro" },
        h("div", { class: "s-head" }, h("div", {}, h("div", { class: "kicker" }, t("scan_kicker")), h("h2", {}, t("scan_intro_title")))),
        h("div", { class: "steps3" }, step(1, icons.box, "scan_step1", "sun"), step(2, icons.vessel, "scan_step2", "magenta"),
          step(3, icons.check, "scan_step3", "green")),
        h("div", { class: "row" }, h("span", { class: "label" }, t("scan_profile")), profs),
        h("div", { class: "bar-actions" },
          h("button", { class: "btn ghost left", onclick: tap(() => api.cancel()) }, svg(icons.back), t("back")),
          h("button", { class: "btn big turq", onclick: tap(() => api.act("start")) }, svg(icons.play), t("scan_start"))));
    },
    update(d) {
      refs.profs && refs.profs.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.dataset.p === d.selected));
    },
    leave() { refs = {}; },
  },

  preflight: { render: (d) => waiting(t("preflight_title"), d.waiting_box ? t("preflight_busy") : t("preflight_sub")) },

  preflight_fail: {
    render(d) {
      return h("section", { class: "center" },
        svg(icons.box, "ico huge warn"), h("h2", {}, t("preflight_fail_title")),
        h("ul", { class: "problems" }, (d.problems || []).map((p) => h("li", {}, p))),
        h("div", { class: "row", style: { justifyContent: "center" } },
          h("button", { class: "btn big sun", onclick: tap(() => api.act("retry")) }, svg(icons.retry), t("preflight_again")),
          h("button", { class: "btn big ghost", onclick: tap(() => api.act("home")) }, svg(icons.home), t("go_home"))));
    },
  },

  stopping: { render: () => waiting(t("stopping_title"), t("stopping_sub")) },

  scanning: {
    render(d, ctx) {
      const plate = h("div", { class: "plate" });
      const hand = h("div", { class: "plate-hand" });
      const big = h("div", { class: "plate-photo" }, svg(icons.vessel));
      const ring = h("div", { class: "turntable" }, plate, hand, big);
      const steps = h("ol", { class: "scan-steps" }, STAGES.map((s) => h("li", { "data-s": s }, svg(STAGE_ICON[s]), t("scan_" + s))));
      const light = h("span", { class: "badge light" }, t("light_none"));
      const temp = h("span", { class: "badge" }, "—");
      const thermal = h("div", { class: "thermal-box" }, h("span", { class: "muted" }, t("thermal_wait")));
      const bar = h("i", {});
      const msg = h("p", { class: "lead msg" }, d.message_es || "");
      const angle = h("b", { class: "angle" }, "0°");
      refs = { plate, hand, big, steps, light, temp, thermal, bar, msg, angle, stage: "" };
      ctx.on("scan_thermal", (m) => refs.thermal.replaceChildren(h("img", { src: m.url, alt: "" })));
      ctx.on("scan_photo", (m) => {
        if (m.kind && m.kind !== "photogrammetry") { big.style.backgroundImage = `url("${m.url}")`; big.replaceChildren(); return; }
        const a = Number(m.angle) || 0, outer = m.camera !== "B";
        const dot = h("div", { class: "shot" + (outer ? "" : " inner") });
        dot.style.setProperty("--a", a + "deg");
        dot.style.backgroundImage = `url("${m.url}")`;
        plate.append(dot);
        big.style.backgroundImage = `url("${m.url}")`;
        big.replaceChildren();
      });
      ctx.on("scan_live", (m) => {
        const det = m.detail || {};
        refs.bar.style.width = pct(m.fraction) + "%";
        refs.msg.textContent = m.message_es || "";
        const stage = m.stage === "analysis" ? "analyzing" : m.stage;
        if (stage === "done") refs.steps.querySelectorAll("li").forEach((li) => { li.className = "done"; });
        else if (stage !== refs.stage) {
          refs.stage = stage;
          const i = STAGES.indexOf(stage);
          refs.steps.querySelectorAll("li").forEach((li, k) => { li.className = k < i ? "done" : k === i ? "now" : ""; });
          document.body.dataset.scanStage = stage || "";
        }
        if (det.platter_deg != null) {
          refs.hand.style.setProperty("--a", det.platter_deg + "deg");
          refs.angle.textContent = Math.round(det.platter_deg) + "°";
        }
        if (det.light !== undefined) { refs.light.textContent = lightName(det.light) || t("light_none"); refs.light.dataset.l = det.light; }
        if (det.temp_max_c != null) refs.temp.textContent = t("temp_max", { c: Number(det.temp_max_c).toFixed(1) });
        if (det.thermal_preview_url) refs.thermal.replaceChildren(h("img", { src: det.thermal_preview_url, alt: "" }));
        if (det.t != null && stage === "thermal") refs.temp.textContent = Math.round(det.t) + " s";
      });
      return h("section", { class: "scanning" },
        h("div", { class: "table-side" }, ring, h("div", { class: "row plate-info" }, angle, light, d.weight_g ? h("span", { class: "badge" }, svg(icons.scale), Math.round(d.weight_g) + " g") : null)),
        h("div", { class: "scan-side" },
          h("div", { class: "kicker" }, t("scan_kicker")), h("h2", {}, t("scanning_title")),
          steps, h("div", { class: "weave" }, bar), msg,
          h("div", { class: "row live" }, h("div", { class: "grow" }, h("div", { class: "label" }, t("thermal_label")), thermal), temp),
          h("div", { class: "bar-actions" },
            h("button", { class: "btn red", onclick: tap(() => api.act("stop_scan")) }, svg(icons.stop), t("scan_stop")))));
    },
    update() { /* live data arrives through scan_live / scan_photo events */ },
    leave() { refs = {}; delete document.body.dataset.scanStage; },
  },
};
