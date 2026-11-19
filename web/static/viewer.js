// 3D world viewer synced with the 2D video. World data is OpenCV (x right, y down, z forward of the first camera);
// for display we flip to y-up: (x, y, z) -> (x, -y, -z).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
               [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]];
const COLORS = { left: 0xff9628, right: 0x3290ff };
const TRAIL = 45;

const v3 = (p) => new THREE.Vector3(p[0], -p[1], -p[2]);

export function startViewer({ data, task, $ }) {
  const T = data.num_frames, fps = data.fps;
  const host = $('scene3d');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  host.append(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1420);
  const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.005, 100);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // camera centres and orientations
  const camPos = [], camRot = [];
  for (let t = 0; t < T; t++) {
    const c = data.cam[t];
    const R = new THREE.Matrix3().set(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
    camRot.push(R); camPos.push(v3([c[9], c[10], c[11]]));
  }

  // scene extent from valid joints + camera path
  const box = new THREE.Box3();
  camPos.forEach(p => box.expandByPoint(p));
  const hands = {};
  for (const h of ['left', 'right']) {
    const H = data.hands[h];
    for (let t = 0; t < T; t++) if (H.valid[t]) H.joints[t].forEach(j => box.expandByPoint(v3(j)));
    // skeleton
    const segGeom = new THREE.BufferGeometry();
    segGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(BONES.length * 6), 3));
    const seg = new THREE.LineSegments(segGeom, new THREE.LineBasicMaterial({ color: COLORS[h] }));
    const dots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.006, 10, 8), new THREE.MeshBasicMaterial({ color: COLORS[h] }), 21);
    const trailGeom = new THREE.BufferGeometry();
    trailGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3));
    const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ color: COLORS[h], transparent: true, opacity: 0.55 }));
    scene.add(seg, dots, trail);
    hands[h] = { H, seg, dots, trail };
  }
  const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
  const ext = Math.max(size.x, size.y, size.z, 0.3);

  // ground grid just below everything, axes at the world origin (first camera)
  const grid = new THREE.GridHelper(ext * 4, 20, 0x3a4458, 0x232a38);
  grid.position.set(centre.x, box.min.y - 0.05, centre.z);
  scene.add(grid, new THREE.AxesHelper(0.08));

  // camera path + frustum
  const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(camPos), new THREE.LineBasicMaterial({ color: 0xbfc7d6 }));
  const fw = data.width / (2 * data.K[0][0]), fh = data.height / (2 * data.K[1][1]), d = Math.max(0.04, ext * 0.08);
  const corners = [[-fw, -fh], [fw, -fh], [fw, fh], [-fw, fh]].map(([x, y]) => [x * d, y * d, d]);
  const frGeom = new THREE.BufferGeometry();
  frGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(16 * 3), 3));
  const frustum = new THREE.LineSegments(frGeom, new THREE.LineBasicMaterial({ color: 0xffffff }));
  scene.add(path, frustum);

  function resetView() {
    const p0 = camPos[0].clone();
    const dir = centre.clone().sub(p0);
    camera.position.copy(p0.clone().sub(dir.clone().multiplyScalar(0.8)).add(new THREE.Vector3(0, ext * 0.6, 0)));
    controls.target.copy(centre);
    controls.update();
  }
  resetView();
  $('resetView').onclick = resetView;

  function camPoint(t, pc) {  // OpenCV camera coords -> world (display)
    const R = camRot[t].elements;  // column-major
    const x = R[0] * pc[0] + R[3] * pc[1] + R[6] * pc[2], y = R[1] * pc[0] + R[4] * pc[1] + R[7] * pc[2], z = R[2] * pc[0] + R[5] * pc[1] + R[8] * pc[2];
    const c = data.cam[t];
    return v3([x + c[9], y + c[10], z + c[11]]);
  }

  const m4 = new THREE.Matrix4();
  function setFrame(t) {
    for (const h of ['left', 'right']) {
      const { H, seg, dots, trail } = hands[h];
      const on = !!H.valid[t];
      seg.visible = dots.visible = on;
      if (on) {
        const P = H.joints[t].map(v3);
        const a = seg.geometry.attributes.position.array;
        BONES.forEach(([i, j], k) => { P[i].toArray(a, k * 6); P[j].toArray(a, k * 6 + 3); });
        seg.geometry.attributes.position.needsUpdate = true;
        P.forEach((p, k) => dots.setMatrixAt(k, m4.makeTranslation(p.x, p.y, p.z)));
        dots.instanceMatrix.needsUpdate = true;
      }
      const ta = trail.geometry.attributes.position.array;
      let n = 0;
      for (let s = Math.max(0, t - TRAIL + 1); s <= t; s++) if (H.valid[s]) { v3(H.joints[s][0]).toArray(ta, 3 * n++); }
      trail.geometry.setDrawRange(0, n);
      trail.geometry.attributes.position.needsUpdate = true;
      trail.visible = $('optTrail').checked && n > 1;
    }
    const o = camPos[t], C = corners.map(c => camPoint(t, c));
    const fa = frustum.geometry.attributes.position.array;
    const pairs = [[o, C[0]], [o, C[1]], [o, C[2]], [o, C[3]], [C[0], C[1]], [C[1], C[2]], [C[2], C[3]], [C[3], C[0]]];
    pairs.forEach(([p, q], k) => { p.toArray(fa, k * 6); q.toArray(fa, k * 6 + 3); });
    frustum.geometry.attributes.position.needsUpdate = true;
    frustum.visible = path.visible = $('optCam').checked;
  }

  // video sync
  const vids = { overlay: $('v_overlay'), input: $('v_input') };
  vids.overlay.src = `/files/${task}/overlay.mp4`;
  vids.input.src = `/files/${task}/input.mp4`;
  let active = vids.overlay;
  document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
    const next = vids[b.dataset.v];
    if (next === active) return;
    next.currentTime = active.currentTime;
    const playing = !active.paused;
    active.pause(); active.hidden = true; next.hidden = false; active = next;
    if (playing) next.play();
    document.querySelectorAll('.tabs button').forEach(x => x.classList.toggle('on', x === b));
  });
  const slider = $('slider'); slider.max = T - 1;
  const frameOf = () => Math.min(T - 1, Math.max(0, Math.floor(active.currentTime * fps + 1e-3)));
  slider.oninput = () => { active.pause(); active.currentTime = (Number(slider.value) + 0.5) / fps; };
  $('play').onclick = () => { if (active.paused) { if (active.ended) active.currentTime = 0; active.play(); } else active.pause(); };
  Object.values(vids).forEach(v => { v.onplay = () => $('play').textContent = '暂停'; v.onpause = () => $('play').textContent = '播放'; });

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);
  resize();

  let last = -1;
  function loop() {
    const t = frameOf();
    if (t !== last) {
      setFrame(t); last = t; slider.value = t;
      const L = data.hands.left.valid[t] ? '左' : '', R = data.hands.right.valid[t] ? '右' : '';
      $('frame').textContent = `${t + 1} / ${T}` + ((L || R) ? '' : ' · 无手');
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  ['optTrail', 'optCam'].forEach(id => $(id).onchange = () => { last = -1; });
  loop();
}
