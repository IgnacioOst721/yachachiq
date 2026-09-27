// 3D model viewer: vendored three.js (MIT) GLTFLoader + OrbitControls. One finger turns, two fingers zoom.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { h } from "../core.js";
import { t } from "../i18n.js";

export async function mountModel(box, url) {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.toneMapping = THREE.NeutralToneMapping;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 50);
  scene.add(new THREE.HemisphereLight(0xfff3d6, 0x251350, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(2, 3, 2);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x00b5cc, 1.2);
  rim.position.set(-3, 1, -2);
  scene.add(rim);

  const gltf = await new GLTFLoader().loadAsync(url);
  const model = gltf.scene;
  const bbox = new THREE.Box3().setFromObject(model);
  const size = bbox.getSize(new THREE.Vector3());
  const center = bbox.getCenter(new THREE.Vector3());
  model.position.sub(center);                                   // centre on the origin
  scene.add(model);
  const radius = bbox.getBoundingSphere(new THREE.Sphere()).radius || 1;
  const disc = new THREE.Mesh(new THREE.CircleGeometry(radius * 1.1, 64),
    new THREE.MeshBasicMaterial({ color: 0xffc20e, transparent: true, opacity: 0.18 }));
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = -size.y / 2 - radius * 0.01;
  scene.add(disc);
  const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.15;
  camera.position.set(0.55, 0.35, 0.75).normalize().multiplyScalar(dist);
  camera.near = radius / 100;
  camera.far = radius * 50;
  camera.updateProjectionMatrix();

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = !still;
  controls.autoRotateSpeed = 2.0;
  controls.minDistance = radius * 1.2;
  controls.maxDistance = dist * 2.5;
  controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
  controls.addEventListener("start", () => { controls.autoRotate = false; });

  box.replaceChildren(renderer.domElement);
  renderer.domElement.style.touchAction = "none";
  const mm = [size.x, size.y, size.z].map((v) => v * 1000);
  if (mm.every((v) => v > 5 && v < 1000)) {                     // glTF is in metres; show it only if plausible
    box.append(h("div", { class: "dims" }, t("model_dims", { w: mm[0].toFixed(0), h: mm[1].toFixed(0), d: mm[2].toFixed(0) })));
  }
  const fit = () => {
    const w = box.clientWidth || 800, hgt = box.clientHeight || 600;
    renderer.setSize(w, hgt, false);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    camera.aspect = w / hgt;
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(fit);
  ro.observe(box);
  fit();
  let raf = 0;
  const loop = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(loop); };
  loop();

  return {
    dispose() {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) [].concat(o.material).forEach((m) => { for (const v of Object.values(m)) if (v && v.isTexture) v.dispose(); m.dispose(); });
      });
      renderer.dispose();
      renderer.forceContextLoss();
    },
  };
}
