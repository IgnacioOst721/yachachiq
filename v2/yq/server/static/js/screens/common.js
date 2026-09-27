// Shared screens: kind error with retry, waiting spinner, unknown-screen fallback.
import { h, svg, api, tap } from "../core.js";
import { chakana, icons } from "../icons.js";
import { t, tm } from "../i18n.js";

export function waiting(title, sub, extra) {
  return h("section", { class: "center wait" },
    h("div", { class: "spinner", html: chakana }),
    h("h2", {}, title), sub ? h("p", { class: "lead" }, sub) : null, extra || null);
}

export function backBtn(action = "back", label) {
  return h("button", { class: "btn ghost", onclick: tap(() => api.act(action)) }, svg(icons.back), label || t("back"));
}

export const screens = {
  error: {
    render(d) {
      return h("section", { class: "center err" },
        h("div", { class: "err-mark" }, svg(icons.sparkle)),
        h("div", { class: "kicker" }, t("err_kicker")),
        h("h2", {}, tm(d.key, d.message_es) || t("err_generic")),
        h("div", { class: "row", style: { justifyContent: "center", marginTop: "2rem" } },
          h("button", { class: "btn big sun", onclick: tap(() => api.act("retry")) }, svg(icons.retry), t("retry")),
          h("button", { class: "btn big ghost", onclick: tap(() => api.act("home")) }, svg(icons.home), t("go_home"))),
        d.detail ? h("p", { class: "err-detail" }, d.detail) : null);
    },
  },
  unknown: {
    render(d) {
      return waiting(t("loading"), d && d.name ? "(" + d.name + ")" : "");
    },
  },
};
