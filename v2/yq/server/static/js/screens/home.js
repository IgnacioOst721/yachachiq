// Home: two big doors (tell a story / analyse an object). Also the attract screen.
import { h, svg, api, tap, toast } from "../core.js";
import { icons } from "../icons.js";
import { t } from "../i18n.js";

async function go(flow, extra) {
  const r = await api.start(flow, extra);
  if (!r.ok && r.status === 409) toast(t("busy"), "error");
}

export const screens = {
  home: {
    render() {
      return h("section", { class: "home" },
        h("div", { class: "home-head" },
          h("div", {}, h("div", { class: "kicker" }, t("home_kicker")), h("h1", {}, t("home_title"))),
          h("p", { class: "lead" }, t("home_lead"))),
        h("div", { class: "choices doors" },
          h("button", { class: "choice magenta door-story", onclick: tap(() => go("story")) },
            h("div", { class: "door-icons" }, svg(icons.mic), svg(icons.hand), svg(icons.keyboard)),
            h("div", { class: "t" }, t("door_story")), h("div", { class: "d" }, t("door_story_d")),
            h("span", { class: "door-go" }, t("start"), svg(icons.play))),
          h("button", { class: "choice turq door-scan", onclick: tap(() => go("scan")) },
            h("div", { class: "door-icons" }, svg(icons.vessel), svg(icons.box), svg(icons.light)),
            h("div", { class: "t" }, t("door_scan")), h("div", { class: "d" }, t("door_scan_d")),
            h("span", { class: "door-go" }, t("start"), svg(icons.play)))),
        h("div", { class: "home-foot" }, h("span", {}, t("home_foot")), h("span", { class: "muted" }, "WRO 2026 · Future Innovators · Colegio FDR · Lima")));
    },
  },
};
