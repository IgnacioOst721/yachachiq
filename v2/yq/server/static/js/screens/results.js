// Scan results: identification card, 3D model, RTI relighting, UV, thermal, measurements.
import { h, svg, api, tap, pct } from "../core.js";
import { icons } from "../icons.js";
import { t } from "../i18n.js";
import { regionLabel } from "./scan_context.js";

const TABS = [["summary", icons.sparkle], ["model", icons.cube], ["rti", icons.light], ["uv", icons.uv],
  ["thermal", icons.thermo], ["measure", icons.ruler]];
let viewer = null, tab = "summary";

function fmt(v, unc, unit) {
  const dec = !unc ? (Number.isInteger(Number(v)) ? 0 : 1) : unc < 0.1 ? 2 : unc < 10 ? 1 : 0;
  const u = unit === "cm3" ? "cm³" : unit === "g/cm3" ? "g/cm³" : unit;
  return Number(v).toFixed(dec) + (unc ? " ± " + Number(unc).toFixed(dec) : "") + " " + u;
}
const confWord = (c) => t(c >= 0.75 ? "conf_high" : c >= 0.5 ? "conf_mid" : "conf_low");

function findings(r, analysis) {
  const fs = (r.findings || []).filter((f) => f.analysis === analysis);
  if (!fs.length) return null;
  return h("div", { class: "findings" }, fs.map((f) => h("div", { class: "finding " + (f.severity || "info") },
    h("b", {}, f.title_es), h("span", {}, f.detail_es))));
}

function summary(r) {
  const id = r.identification;
  const photo = r.viewers.photo ? h("div", { class: "paper photo" }, h("img", { src: r.viewers.photo, alt: "" })) : null;
  if (!id) return h("div", { class: "res-grid" }, photo, h("div", { class: "panel" }, h("h3", {}, t("id_none"))));
  const c = Number(id.confidence) || 0;
  const main = h("div", { class: "id-main" },
    h("div", { class: "label" }, t("id_kicker")),
    h("h2", {}, id.object_type_es || id.object_type),
    h("div", { class: "row" }, h("span", { class: "badge culture" }, id.culture), h("span", { class: "badge" }, id.period),
      id.region ? h("span", { class: "badge" }, id.region) : null),
    h("p", {}, h("b", {}, t("id_material") + ": "), id.material_es || id.material),
    h("div", { class: "conf" }, h("span", {}, t("id_conf")), h("div", { class: "meter" }, h("i", { style: { width: pct(c) + "%" } })),
      h("b", {}, pct(c) + "% · " + confWord(c))),
    id.description_es ? h("p", { class: "desc" }, id.description_es) : null);
  const more = h("div", { class: "id-more" }, placeBox(r.context || {}, id),
    h("div", { class: "label" }, t("id_evidence")), h("ul", {}, (id.evidence || []).map((e) => h("li", {}, e))),
    (id.alternatives || []).length ? h("div", { class: "label" }, t("id_alternatives")) : null,
    h("ul", { class: "alts" }, (id.alternatives || []).map((a) =>
      h("li", {}, h("b", {}, a.culture + " (" + pct(a.confidence) + "%)"), a.period ? " · " + a.period : "", h("br"), h("small", {}, a.why || "")))),
    (id.similar || []).length ? h("div", { class: "label" }, t("id_similar")) : null,
    (id.similar || []).length ? h("div", { class: "similar" }, id.similar.slice(0, 3).map((x) => h("div", { class: "sim" },
      x.image && x.image.startsWith("/") ? h("img", { src: x.image, alt: "" }) : null,
      h("b", {}, x.title), h("small", {}, [x.culture, x.date, x.museum].filter(Boolean).join(" · "))))) : null);
  return h("div", { class: "res-grid summary" + (photo ? "" : " no-photo") }, photo, h("div", { class: "id-card panel" }, main, more));
}

// the visitor's place is a clue: show it, what it changed, and the image-only answer when it differs
function placeBox(ctx, id) {
  const place = [ctx.found_where, regionLabel(ctx.region_hint)].filter(Boolean).join(" · ");
  const io = id.image_only;
  const differs = io && io.culture && io.culture !== id.culture;
  if (!place && !id.context_effect_es && !differs) return null;
  return h("div", { class: "ctx-box" },
    place ? h("p", {}, h("b", {}, t("id_place") + ": "), place) : null,
    id.context_effect_es ? h("p", { class: "effect" }, id.context_effect_es) : null,
    differs ? h("p", { class: "compare" }, t("id_compare", { a: io.culture + " (" + pct(io.confidence) + "%)",
      b: id.culture + " (" + pct(id.confidence) + "%)" })) : null);
}

function imagePair(a, b, la, lb, extra) {
  return h("div", { class: "res-grid" },
    a ? h("figure", { class: "paper" }, h("img", { src: a, alt: "" }), h("figcaption", {}, la)) : null,
    b ? h("figure", { class: "paper" }, h("img", { src: b, alt: "" }), h("figcaption", {}, lb)) : null,
    extra);
}

function measures(r) {
  const rows = r.measurements || [];
  if (!rows.length) return h("p", { class: "lead" }, t("meas_none"));
  return h("div", { class: "panel meas" }, h("table", {}, h("tbody", {}, rows.map((m) => h("tr", {},
    h("th", {}, t("meas_" + m.name) === "meas_" + m.name ? m.name : t("meas_" + m.name)),
    h("td", { class: "val" }, fmt(m.value, m.uncertainty, m.unit)),
    h("td", { class: "muted" }, [m.method, m.note].filter(Boolean).join(" · ")))))),
    h("p", { class: "fine" }, t("meas_note")));
}

async function mountViewer(kind, box, url) {
  box.replaceChildren(h("p", { class: "lead muted" }, t("loading")));
  try {
    if (kind === "model") viewer = await (await import("../viewers/model3d.js")).mountModel(box, url);
    else viewer = await (await import("../viewers/rti.js")).mountRti(box, url);
  } catch (e) {
    console.error(e);
    box.replaceChildren(h("p", { class: "lead" }, t("viewer_failed")));
  }
}

function body(r, name) {
  if (viewer && viewer.dispose) { viewer.dispose(); viewer = null; }
  const v = r.viewers;
  if (name === "summary") return summary(r);
  if (name === "measure") return measures(r);
  if (name === "uv") return imagePair(v.uv.visible, v.uv.uv, t("uv_visible"), t("uv_uv"),
    h("div", { class: "side-notes" }, v.uv.overlay ? h("figure", { class: "paper" }, h("img", { src: v.uv.overlay, alt: "" }), h("figcaption", {}, t("uv_overlay"))) : null,
      findings(r, "uv"), h("p", { class: "fine" }, t("uv_explain"))));
  if (name === "thermal") return imagePair(v.thermal.max, v.thermal.anomaly, t("th_max"), t("th_anomaly"),
    h("div", { class: "side-notes" }, h("div", { class: "iron-scale" }, h("span", {}, t("th_cold")), h("i", {}), h("span", {}, t("th_hot"))),
      findings(r, "thermal"), h("p", { class: "fine" }, t("th_explain"))));
  const url = name === "model" ? v.model : v.rti;
  if (!url) return h("p", { class: "lead" }, t("viewer_missing"));
  const box = h("div", { class: "viewer-box " + name });
  setTimeout(() => mountViewer(name, box, url), 30);
  return h("div", { class: "viewer-wrap" }, box, h("div", { class: "side-notes" },
    h("p", { class: "lead" }, t(name === "model" ? "model_hint" : "rti_hint")), name === "rti" ? findings(r, "rti") : null,
    h("p", { class: "fine" }, t(name === "model" ? "model_explain" : "rti_explain"))));
}

export const screens = {
  results: {
    render(d) {
      const r = d.result || {};
      tab = "summary";
      const content = h("div", { class: "tab-body" });
      const avail = { model: !!r.viewers?.model, rti: !!r.viewers?.rti, uv: !!(r.viewers?.uv?.uv), thermal: !!(r.viewers?.thermal?.max || r.viewers?.thermal?.anomaly) };
      const tabs = h("div", { class: "tabs", role: "tablist" }, TABS.filter(([k]) => avail[k] !== false).map(([k, ic]) =>
        h("button", { role: "tab", "data-t": k, "aria-selected": String(k === tab), onclick: tap(() => select(k), 250) }, svg(ic), t("tab_" + k))));
      const select = (k) => {
        tab = k;
        tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.t === k)));
        content.replaceChildren(body(r, k));
      };
      content.append(body(r, tab));
      return h("section", { class: "results" },
        h("div", { class: "s-head" }, h("div", {}, h("div", { class: "kicker" }, t("results_kicker")), h("h2", {}, t("results_title"))),
          h("div", { class: "row head-actions" }, r.mock ? h("span", { class: "badge sim" }, t("simulated")) : null,
            h("button", { class: "btn ghost", onclick: tap(() => api.act("finish")) }, svg(icons.home), t("finish")),
            h("button", { class: "btn turq", onclick: tap(() => api.act("new_scan")) }, svg(icons.vessel), t("new_scan")))),
        tabs, content);
    },
    leave() { if (viewer && viewer.dispose) viewer.dispose(); viewer = null; },
  },
};
