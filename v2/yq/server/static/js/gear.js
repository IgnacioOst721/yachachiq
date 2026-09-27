// Gear menu for the team: password first (config.KIOSK_EXIT_PASSWORD), then status, emergency exit,
// publish now, reprint, and (mock only) "simulate a covered camera".
import { h, svg, api, tap, toast } from "./core.js";
import { icons } from "./icons.js";
import { t } from "./i18n.js";
import { openKeyboard, closeKeyboard } from "./keyboard.js";

let pw = "";

function close(ov) { closeKeyboard(); ov.remove(); }

export function openGear() {
  const ov = h("div", { class: "overlay", id: "gear" });
  const sheet = h("div", { class: "sheet gear-sheet" });
  ov.append(sheet);
  document.body.append(ov);
  askPassword(ov, sheet);
}

function askPassword(ov, sheet) {
  let typed = "";
  const field = h("button", { class: "chip pw", onclick: tap(() => kb()) }, t("gear_pw_tap"));
  const enter = async (v) => {
    const r = await api.post("/api/admin/check", { password: v });
    if (!r.ok) { toast(t("gear_pw_bad"), "error"); typed = ""; field.textContent = t("gear_pw_tap"); return; }
    pw = v;
    closeKeyboard();
    panel(ov, sheet, r.data.status);
  };
  const kb = () => openKeyboard({ mode: "password", text: typed, title: t("gear_pw"), doneLabel: t("gear_enter"),
    onChange: (v) => { typed = v; field.textContent = "•".repeat(v.length) || t("gear_pw_tap"); }, onDone: enter });
  sheet.replaceChildren(h("h2", {}, t("gear_title")), h("p", { class: "lead" }, t("gear_pw_lead")), field,
    h("div", { class: "bar-actions" }, h("button", { class: "btn", onclick: tap(() => close(ov)) }, svg(icons.x), t("kb_close"))));
  kb();
}

function statusList(st) {
  const s = st.subsystems || {}, r = st.reachable || {}, p = st.publish || {};
  const row = (k, v, good) => h("tr", {}, h("th", {}, k), h("td", { class: good === true ? "good" : good === false ? "bad" : "" }, String(v ?? "—")));
  return h("table", { class: "gear-status" }, h("tbody", {},
    ["voice", "languages", "sign", "art", "box", "camera"].map((k) => row(k, s[k], !String(s[k] || "").startsWith("mock") ? true : null)),
    row("mac", r.mac ? "ok" : "no responde", !!r.mac),
    row("printer", (s.printer || "") + " · " + (r.printer ? "ok" : "no responde"), !!r.printer),
    row("hologram", (s.hologram || "") + " · " + (r.hologram ? "ok" : "no responde"), !!r.hologram),
    row("web", `${p.mode} · pendientes ${p.pending ?? "?"} · publicadas ${p.published ?? "?"} · último: ${(p.last || {}).status || "—"}`),
    Object.keys(s.errors || {}).length ? row("errores", JSON.stringify(s.errors)) : null));
}

async function panel(ov, sheet, st) {
  const listBox = h("div", { class: "gear-stories" }, h("p", { class: "muted" }, t("loading")));
  const mockCam = st.subsystems && st.subsystems.camera === "mock:ui";
  let covered = false;
  const camBtn = mockCam ? h("button", { class: "btn", onclick: tap(async () => {
    covered = !covered;
    await api.post("/api/admin/mock_camera", { password: pw, covered });
    camBtn.lastChild.textContent = covered ? t("gear_cam_open") : t("gear_cam_cover");
  }) }, svg(icons.hand), h("span", {}, t("gear_cam_cover"))) : null;
  const statusBox = h("div", {}, statusList(st));
  sheet.replaceChildren(
    h("h2", {}, t("gear_title")),
    statusBox,
    h("div", { class: "row gear-actions" },
      h("button", { class: "btn red", onclick: tap(async () => {
        const r = await api.post("/api/kiosk/exit", { password: pw });
        toast(r.ok ? t("gear_exit_ok") : t("gear_pw_bad"), r.ok ? "info" : "error");
      }) }, svg(icons.x), t("gear_exit")),
      h("button", { class: "btn sun", onclick: tap(async () => {
        const r = await api.post("/api/admin/publish", { password: pw });
        toast(t("gear_publish_res", { s: (r.data && r.data.status) || "error" }));
      }) }, svg(icons.globe), t("gear_publish")),
      h("button", { class: "btn", onclick: tap(async () => {
        const r = await api.post("/api/admin/refresh", { password: pw });
        if (r.ok) statusBox.replaceChildren(statusList(r.data));
      }) }, svg(icons.retry), t("gear_refresh")),
      h("button", { class: "btn", onclick: () => location.reload() }, svg(icons.retry), t("gear_reload")),
      camBtn),
    h("h3", {}, t("gear_reprint")), listBox,
    h("div", { class: "bar-actions" }, h("button", { class: "btn big", onclick: tap(() => close(ov)) }, svg(icons.x), t("kb_close"))));
  const r = await api.post("/api/admin/stories", { password: pw });
  const rows = (r.data && r.data.stories) || [];
  listBox.replaceChildren(...(rows.length ? rows.slice(0, 8).map((s) => h("div", { class: "row story-row" },
    h("b", {}, s.title || s.id), h("span", { class: "badge" }, s.state), h("span", { class: "muted" }, t("printer") + ": " + (s.printer || "—")),
    s.has_drawing ? h("button", { class: "chip", onclick: tap(async () => {
      const x = await api.post("/api/admin/reprint", { password: pw, story_id: s.id });
      toast(x.ok ? t("gear_reprint_ok") : (x.data.error || "error"), x.ok ? "info" : "error");
    }) }, svg(icons.printer), t("gear_reprint_btn")) : null)) : [h("p", { class: "muted" }, t("gear_no_stories"))]));
}
