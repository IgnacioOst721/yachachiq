// RTI relighting viewer (WebGL): drag a finger to move the light over the object.
// Data format "yq-ptm-lrgb-v1" (docs/ui.md): ptm.json + 2 RGB PNGs with the 6 PTM coefficients + albedo PNG.
//   a_i = bias[i] + scale[i] * byte/255 ;  L = a0 lu² + a1 lv² + a2 lu lv + a3 lu + a4 lv + a5 ;  colour = albedo * L
import { h, tap } from "../core.js";
import { t } from "../i18n.js";

const VS = `attribute vec2 p; varying vec2 uv; void main(){ uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;
const FS = `precision mediump float;
uniform sampler2D c0, c1, alb; uniform vec3 sA, sB, bA, bB; uniform vec2 light; uniform float gain; uniform int mode;
varying vec2 uv;
void main(){
  vec3 A = texture2D(c0, uv).rgb * sA + bA;
  vec3 B = texture2D(c1, uv).rgb * sB + bB;
  float lu = light.x, lv = light.y;
  float L = A.x*lu*lu + A.y*lv*lv + A.z*lu*lv + B.x*lu + B.y*lv + B.z;
  L = max(L, 0.0) * gain;
  vec3 base = mode == 1 ? vec3(0.86, 0.84, 0.8) : texture2D(alb, uv).rgb;
  gl_FragColor = vec4(min(base * L, vec3(1.0)), 1.0);
}`;

function loadImage(src) {
  return new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error(src)); im.src = src; });
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export async function mountRti(box, url) {
  const meta = await (await fetch(url, { cache: "no-cache" })).json();
  if (meta.format !== "yq-ptm-lrgb-v1") throw new Error("unknown RTI format " + meta.format);
  const base = url.slice(0, url.lastIndexOf("/") + 1);
  const [i0, i1, ia] = await Promise.all([meta.coeff_images[0], meta.coeff_images[1], meta.albedo].map((n) => loadImage(base + n)));
  const canvas = h("canvas", { class: "rti-canvas" });
  const sun = h("div", { class: "rti-sun" });
  const gl = canvas.getContext("webgl", { preserveDrawingBuffer: false });
  if (!gl) throw new Error("no WebGL");
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);                         // image top = uv.y 1
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);        // coefficients are data, not colours
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  const textures = [i0, i1, ia].map((img, k) => {
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + k);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
    return tex;
  });
  const U = (n) => gl.getUniformLocation(prog, n);
  gl.uniform1i(U("c0"), 0); gl.uniform1i(U("c1"), 1); gl.uniform1i(U("alb"), 2);
  gl.uniform3fv(U("sA"), meta.scale.slice(0, 3)); gl.uniform3fv(U("sB"), meta.scale.slice(3, 6));
  gl.uniform3fv(U("bA"), meta.bias.slice(0, 3)); gl.uniform3fv(U("bB"), meta.bias.slice(3, 6));
  let light = [-0.6, 0.6], mode = 0, gain = 1.15, auto = !matchMedia("(prefers-reduced-motion: reduce)").matches, raf = 0;

  const draw = () => {
    const w = canvas.clientWidth, hh = canvas.clientHeight, dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(hh * dpr)) {
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(hh * dpr);
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform2f(U("light"), light[0], light[1]);
    gl.uniform1f(U("gain"), mode ? gain * 1.25 : gain);
    gl.uniform1i(U("mode"), mode);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const r = Math.min(w, hh) / 2;
    sun.style.transform = `translate(${w / 2 + light[0] * r}px, ${hh / 2 - light[1] * r}px)`;
  };
  const setFrom = (ev) => {
    const b = canvas.getBoundingClientRect(), r = Math.min(b.width, b.height) / 2;
    let x = (ev.clientX - b.left - b.width / 2) / r, y = -(ev.clientY - b.top - b.height / 2) / r;
    const n = Math.hypot(x, y);
    if (n > 0.97) { x *= 0.97 / n; y *= 0.97 / n; }
    light = [x, y];
    draw();
  };
  canvas.addEventListener("pointerdown", (e) => { auto = false; canvas.setPointerCapture(e.pointerId); setFrom(e); });
  canvas.addEventListener("pointermove", (e) => { if (canvas.hasPointerCapture(e.pointerId)) setFrom(e); });
  const t0 = performance.now();
  const spin = () => {
    if (auto) { const a = (performance.now() - t0) / 1600; light = [Math.cos(a) * 0.85, Math.sin(a) * 0.85]; draw(); }
    raf = requestAnimationFrame(spin);
  };
  const modes = h("div", { class: "langs rti-modes" }, [["rti_color", 0], ["rti_relief", 1]].map(([k, m]) =>
    h("button", { "aria-pressed": String(m === mode), onclick: tap((e) => { mode = m; draw();
      modes.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(i === m))); }, 150) }, t(k))));
  const frame = h("div", { class: "rti-frame", style: { aspectRatio: meta.width + " / " + meta.height } }, canvas, sun);
  box.replaceChildren(frame, modes);
  const ro = new ResizeObserver(draw);
  ro.observe(frame);
  draw();
  spin();
  return {
    dispose() {
      cancelAnimationFrame(raf); ro.disconnect();
      textures.forEach((x) => gl.deleteTexture(x)); gl.deleteBuffer(buf); gl.deleteProgram(prog);
      const ext = gl.getExtension("WEBGL_lose_context"); if (ext) ext.loseContext();
    },
  };
}
