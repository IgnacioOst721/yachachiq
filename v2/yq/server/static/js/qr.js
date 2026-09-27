// QR code as crisp SVG (vendored qrcode-generator, MIT). Offline.
import qrcode from "../vendor/qrcode/qrcode.mjs";

export function qrSvg(text, { dark = "#160a31", light = "#fffaf0", margin = 2 } = {}) {
  const qr = qrcode(0, "M");
  qr.addData(text);
  qr.make();
  const n = qr.getModuleCount();
  const size = n + margin * 2;
  let d = "";
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) if (qr.isDark(r, c)) d += `M${c + margin} ${r + margin}h1v1h-1z`;
  }
  return `<svg viewBox="0 0 ${size} ${size}" shape-rendering="crispEdges" role="img" aria-label="QR">` +
    `<rect width="${size}" height="${size}" fill="${light}"/><path d="${d}" fill="${dark}"/></svg>`;
}
