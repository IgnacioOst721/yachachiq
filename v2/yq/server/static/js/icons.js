// Inline SVG icons (no emoji fonts needed on the Jetson). 48x48 grid, thick strokes, currentColor.
const S = (body, extra = "") =>
  `<svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" ${extra}>${body}</svg>`;

export const chakana = `<svg viewBox="0 0 48 48"><path fill="#ffc20e" stroke="#160a31" stroke-width="2.2" stroke-linejoin="round"
  d="M18 3h12v7h8v8h7v12h-7v8h-8v7H18v-7h-8v-8H3V18h7v-8h8z"/><path fill="#e6007e" d="M20 11h8v5h4v4h5v8h-5v4h-4v5h-8v-5h-4v-4h-5v-8h5v-4h4z"/>
  <circle cx="24" cy="24" r="5.2" fill="#160a31"/><circle cx="24" cy="24" r="2.4" fill="#00b5cc"/></svg>`;

export const icons = {
  mic: S('<rect x="17" y="5" width="14" height="24" rx="7"/><path d="M10 22a14 14 0 0 0 28 0M24 36v7M16 43h16"/>'),
  hand: S('<path d="M16 26V11a3 3 0 0 1 6 0v12M22 22V8a3 3 0 0 1 6 0v14M28 22V10a3 3 0 0 1 6 0v16M34 24v-8a3 3 0 0 1 6 0v12c0 9-6 16-15 16-6 0-9-3-13-9l-5-8a3 3 0 0 1 5-3l3 4"/>'),
  keyboard: S('<rect x="3" y="12" width="42" height="26" rx="4"/><path d="M10 20h2M17 20h2M24 20h2M31 20h2M38 20h0M10 26h2M17 26h2M24 26h2M31 26h2M15 32h18"/>'),
  vessel: S('<path d="M17 43h14M14 43c-5-4-7-9-6-15 1-5 5-8 8-9V13M34 43c5-4 7-9 6-15-1-5-5-8-8-9V13M16 13c0-8 16-8 16 0M24 5v6"/><path d="M11 29h26M12 34h24" stroke-dasharray="3 3"/>'),
  box: S('<path d="M6 16l18-9 18 9v18l-18 9-18-9z"/><path d="M6 16l18 9 18-9M24 25v18"/><circle cx="31" cy="30" r="2.5" fill="currentColor"/>'),
  back: S('<path d="M28 10L14 24l14 14"/>'),
  home: S('<path d="M6 22L24 7l18 15M11 19v21h26V19"/><path d="M20 40V28h8v12"/>'),
  gear: S('<path d="M20.5 4h7l1.2 5.2 3.6 1.5 4.5-2.9 5 5-2.9 4.5 1.5 3.6 5.2 1.2v7l-5.2 1.2-1.5 3.6 2.9 4.5-5 5-4.5-2.9-3.6 1.5L27.5 44h-7l-1.2-5.2-3.6-1.5-4.5 2.9-5-5 2.9-4.5-1.5-3.6L2.4 27.5v-7l5.2-1.2 1.5-3.6-2.9-4.5 5-5 4.5 2.9 3.6-1.5z" stroke-width="3"/><circle cx="24" cy="24" r="6.5"/>'),
  check: S('<path d="M8 25l10 10L40 13"/>'),
  x: S('<path d="M12 12l24 24M36 12L12 36"/>'),
  camera: S('<path d="M5 15h9l3-5h14l3 5h9v24H5z"/><circle cx="24" cy="26" r="8"/>'),
  printer: S('<path d="M13 18V6h22v12M13 34H6V18h36v16h-7"/><path d="M13 28h22v14H13z"/><path d="M18 34h12M18 38h8"/>'),
  holo: S('<path d="M8 40h32M14 40l4-8h12l4 8"/><path d="M24 30V8M14 12l10-6 10 6M14 12v12l10 6 10-6V12" stroke-dasharray="4 3"/>'),
  sparkle: S('<path d="M24 4v10M24 34v10M4 24h10M34 24h10M11 11l6 6M31 31l6 6M11 37l6-6M31 17l6-6"/>'),
  search: S('<circle cx="21" cy="21" r="13"/><path d="M31 31l12 12"/>'),
  backspace: S('<path d="M16 10h27v28H16L4 24z"/><path d="M23 18l12 12M35 18L23 30"/>'),
  space: S('<path d="M6 22v10h36V22"/>'),
  trash: S('<path d="M8 12h32M19 12V7h10v5M11 12l3 30h20l3-30"/>'),
  stop: S('<rect x="11" y="11" width="26" height="26" rx="4" fill="currentColor"/>'),
  play: S('<path d="M15 9l24 15-24 15z" fill="currentColor"/>'),
  skip: S('<path d="M10 10l18 14-18 14zM36 10v28"/>'),
  retry: S('<path d="M38 24a14 14 0 1 1-4-10M38 8v8h-8"/>'),
  pencil: S('<path d="M9 39l4-12L33 7l8 8-20 20z"/><path d="M28 12l8 8M9 39l12-4"/>'),
  globe: S('<circle cx="24" cy="24" r="19"/><path d="M5 24h38M24 5c6 6 8 12 8 19s-2 13-8 19c-6-6-8-12-8-19s2-13 8-19z"/>'),
  light: S('<path d="M17 30c-4-3-6-7-6-11a13 13 0 0 1 26 0c0 4-2 8-6 11v6H17z"/><path d="M19 42h10"/>'),
  uv: S('<circle cx="24" cy="24" r="8"/><path d="M24 4v6M24 38v6M4 24h6M38 24h6M10 10l4 4M34 34l4 4M10 38l4-4M34 14l4-4"/>'),
  thermo: S('<path d="M20 8a4 4 0 0 1 8 0v20a9 9 0 1 1-8 0z"/><path d="M24 20v14"/>'),
  cube: S('<path d="M24 4l17 10v20L24 44 7 34V14z"/><path d="M7 14l17 10 17-10M24 24v20"/>'),
  ruler: S('<path d="M5 33L33 5l10 10-28 28z"/><path d="M12 26l4 4M17 21l3 3M22 16l4 4M27 11l3 3"/>'),
  qr: S('<path d="M6 6h14v14H6zM28 6h14v14H28zM6 28h14v14H6zM28 28h6v6h-6zM38 38h4v4h-4zM34 34h4v4"/>'),
  scale: S('<path d="M8 40h32M12 40l4-18h16l4 18M24 22V10M14 10h20"/>'),
};
