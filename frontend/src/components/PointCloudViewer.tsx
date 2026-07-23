import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

/**
 * Renders exactly what the model receives: 2,048 raw points.
 *
 * No surfacing, no mesh — the claim of this project is that drag can be read
 * from this sparse a description of a shape, and the audience should be able to
 * verify that with their own eyes.
 */
interface PointCloudViewerProps {
  points: number[][] | null;
  busy?: boolean;
  /** 있으면 이 메시를 먼저 띄웠다가 점군으로 흩어지는 인트로를 재생한다. */
  meshUrl?: string;
  /** 밀도 애니메이션을 멈추고 이 개수로 고정한다(추론 결과를 볼 때). */
  freezeAt?: number | null;
  onDensityChange?: (count: number, pointsVisible: boolean) => void;
}

const FOV = 34;
const MIN_POINTS = 96;
// 한 바퀴: 차체 → 흩어짐 → 밀도 호흡 → 다시 뭉침. 무한 반복한다.
const LOOP_MS = 13000;
// 사이클 안에서의 구간 경계(비율).
const HOLD_MESH = 0.07;   // 차체를 온전히 보여주는 구간
const SCATTER_END = 0.24; // 흩어짐이 끝나는 지점
const BREATHE_END = 0.76; // 점군 상태로 밀도가 오르내리는 구간
const GATHER_END = 0.93;  // 다시 뭉침이 끝나는 지점
const SMOOTH = (u: number) => u * u * (3 - 2 * u);
// 한 번 차오르고 빠지는 데 걸리는 시간. 너무 빠르면 산만하고 느리면 지루하다.
const CYCLE_MS = 7200;

export function PointCloudViewer({
  points,
  busy = false,
  meshUrl,
  freezeAt = null,
  onDensityChange,
}: PointCloudViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const cloudRef = useRef<THREE.Points | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const spinRef = useRef(true);
  // Framing is derived from the cloud itself so every body type fills the frame.
  const fitRef = useRef({ cx: 1.5, cy: 0, cz: 0.6, spanXY: 4.8, spanZ: 1.4 });
  const totalRef = useRef(0);
  const freezeRef = useRef<number | null>(null);
  const notifyRef = useRef<((count: number, visible: boolean) => void) | undefined>(undefined);
  const lastNotifiedRef = useRef(0);
  // 고정 지점으로 뚝 끊지 않고 흘러들어가게 한다.
  const countRef = useRef(0);
  const meshRef = useRef<THREE.Group | null>(null);
  const meshMatsRef = useRef<THREE.MeshStandardMaterial[]>([]);
  // 메시가 준비된 시점부터 사이클을 센다.
  const loopStartRef = useRef<number | null>(null);
  const meshReadyRef = useRef(false);

  freezeRef.current = freezeAt;
  notifyRef.current = onDensityChange;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 200);
    camera.up.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // The canvas must be laid out by CSS, not by its intrinsic pixel buffer —
    // otherwise a devicePixelRatio of 2 makes it twice its container.
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    host.appendChild(renderer.domElement);

    const grid = new THREE.GridHelper(14, 28, 0x2b3a44, 0x1a242b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    // 메시를 보여줄 때만 필요한 조명. 점군은 조명을 받지 않는다.
    scene.add(new THREE.HemisphereLight(0xdfeaf2, 0x0b1216, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(4, 6, 8);
    scene.add(key);

    let disposed = false;
    if (meshUrl) {
      new GLTFLoader()
        .loadAsync(meshUrl)
        .then((gltf) => {
          if (disposed) return;
          const group = gltf.scene;
          const mats: THREE.MeshStandardMaterial[] = [];
          group.traverse((node) => {
            const mesh = node as THREE.Mesh;
            if (!mesh.isMesh) return;
            const material = new THREE.MeshStandardMaterial({
              color: 0x9fb4c2,
              roughness: 0.42,
              metalness: 0.05,
              transparent: true,
              opacity: 1,
            });
            mesh.material = material;
            mats.push(material);
          });
          meshMatsRef.current = mats;
          meshRef.current = group;
          scene.add(group);
          meshReadyRef.current = true;
          loopStartRef.current = performance.now();
        })
        .catch(() => {
          // 메시를 못 받아도 데모는 점군만으로 성립한다.
          meshReadyRef.current = false;
        });
    }

    let frame = 0;
    let dragging = false;
    let lastX = 0;
    let yaw = 0.72;

    const onDown = (event: PointerEvent) => {
      dragging = true;
      spinRef.current = false;
      lastX = event.clientX;
      renderer.domElement.setPointerCapture(event.pointerId);
    };
    const onMove = (event: PointerEvent) => {
      if (!dragging) return;
      yaw += (event.clientX - lastX) * 0.008;
      lastX = event.clientX;
    };
    const onUp = () => {
      dragging = false;
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerup", onUp);
    renderer.domElement.addEventListener("pointerleave", onUp);

    const resize = () => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const started = performance.now();

    const tick = () => {
      frame = requestAnimationFrame(tick);
      if (spinRef.current) yaw += 0.0026;

      // 점을 매 프레임 다시 만들지 않는다. 지오메트리는 그대로 두고 그리는
      // 개수만 바꾼다 — FPS 순서라 앞에서 n개가 곧 FPS-n 샘플이다.
      // 차체 ↔ 점군을 오가는 한 바퀴. 메시가 옅어지는 만큼 점이 차오르고,
      // 점군 상태에서는 밀도가 오르내리다가, 다시 메시로 뭉친다.
      const cloud = cloudRef.current;
      const total = totalRef.current;
      if (cloud && total > 0) {
        let meshAmount = 0;   // 1이면 차체만, 0이면 점군만
        let densityAmount = 1; // 그릴 점의 비율

        if (freezeRef.current) {
          // 결과가 떠 있는 동안에는 사이클을 멈춘다. 답을 읽는 중에 화면이
          // 계속 변하면 눈이 결과에 머물지 못한다.
          const target = Math.min(freezeRef.current, total);
          countRef.current += (target - countRef.current) * 0.07;
          if (Math.abs(target - countRef.current) < 2) countRef.current = target;
          densityAmount = -1; // countRef를 그대로 쓴다는 표시
        } else if (meshReadyRef.current && loopStartRef.current !== null) {
          const phase = ((performance.now() - loopStartRef.current) % LOOP_MS) / LOOP_MS;
          if (phase < HOLD_MESH) {
            meshAmount = 1;
            densityAmount = 0;
          } else if (phase < SCATTER_END) {
            const k = SMOOTH((phase - HOLD_MESH) / (SCATTER_END - HOLD_MESH));
            meshAmount = 1 - k;
            densityAmount = k;
          } else if (phase < BREATHE_END) {
            const u = (phase - SCATTER_END) / (BREATHE_END - SCATTER_END);
            // 양 끝이 최대 밀도라 앞뒤 크로스페이드와 매끄럽게 이어진다.
            densityAmount = 0.12 + 0.88 * 0.5 * (1 + Math.cos(2 * Math.PI * u));
          } else if (phase < GATHER_END) {
            const k = SMOOTH((phase - BREATHE_END) / (GATHER_END - BREATHE_END));
            meshAmount = k;
            densityAmount = 1 - k;
          } else {
            meshAmount = 1;
            densityAmount = 0;
          }
        } else {
          // 메시가 없으면 점군만으로 호흡한다.
          const phase = ((performance.now() - started) % CYCLE_MS) / CYCLE_MS;
          const triangle = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
          densityAmount = SMOOTH(triangle);
        }

        if (densityAmount >= 0) {
          countRef.current = MIN_POINTS + (total - MIN_POINTS) * densityAmount;
        }
        const count = Math.max(1, Math.round(countRef.current));
        cloud.geometry.setDrawRange(0, count);
        (cloud.material as THREE.PointsMaterial).opacity = 0.92 * (1 - meshAmount);

        if (meshRef.current) {
          meshRef.current.visible = meshAmount > 0.002;
          meshMatsRef.current.forEach((material) => {
            material.opacity = meshAmount;
          });
        }

        const pointsVisible = meshAmount < 0.5;
        const settled =
          freezeRef.current != null && count === Math.min(freezeRef.current, total);
        const changedEnough = Math.abs(count - lastNotifiedRef.current) > total * 0.02;
        if (settled ? lastNotifiedRef.current !== count : pointsVisible && changedEnough) {
          lastNotifiedRef.current = count;
          notifyRef.current?.(count, pointsVisible);
        }
        if (!pointsVisible && lastNotifiedRef.current !== 0) {
          lastNotifiedRef.current = 0;
          notifyRef.current?.(0, false);
        }
      }

      const { cx, cy, cz, spanXY, spanZ } = fitRef.current;
      // Fit the actual extents, not a bounding sphere. A car is long and flat,
      // so a sphere pads it out and leaves the body small in frame.
      const vFov = (FOV * Math.PI) / 180;
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
      const distH = spanXY / 2 / Math.tan(hFov / 2);
      const distV = (spanZ / 2) * 2.1 / Math.tan(vFov / 2);
      const distance = Math.max(distH, distV) * 1.08;

      camera.position.set(
        cx + Math.cos(yaw) * distance * 0.94,
        cy + Math.sin(yaw) * distance * 0.94,
        cz + distance * 0.3,
      );
      camera.lookAt(cx, cy, cz);
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      meshRef.current?.traverse((node) => {
        const mesh = node as THREE.Mesh;
        if (mesh.isMesh) mesh.geometry.dispose();
      });
      meshMatsRef.current.forEach((material) => material.dispose());
      meshMatsRef.current = [];
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerup", onUp);
      renderer.domElement.removeEventListener("pointerleave", onUp);
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      cloudRef.current?.geometry.dispose();
      (cloudRef.current?.material as THREE.Material | undefined)?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      host.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (cloudRef.current) {
      scene.remove(cloudRef.current);
      cloudRef.current.geometry.dispose();
      (cloudRef.current.material as THREE.Material).dispose();
      cloudRef.current = null;
    }
    if (!points?.length) return;

    const positions = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i += 1) {
      positions[i * 3] = points[i][0];
      positions[i * 3 + 1] = points[i][1];
      positions[i * 3 + 2] = points[i][2];
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.computeBoundingBox();

    const box = geometry.boundingBox;
    if (box) {
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      fitRef.current = {
        cx: center.x,
        cy: center.y,
        cz: center.z * 0.8,
        // 카메라가 계속 도므로 어느 각도에서든 가로로 들어와야 한다.
        // 최악은 대각선이 화면 가로와 나란해지는 순간이다.
        spanXY: Math.hypot(size.x, size.y),
        spanZ: size.z,
      };
    }

    const material = new THREE.PointsMaterial({
      size: 0.042,
      color: 0xdbe7ef,
      sizeAttenuation: true,
      transparent: true,
      // 인트로가 남아 있으면 보이지 않게 시작해 메시가 옅어지며 드러난다.
      opacity: meshReadyRef.current ? 0 : 0.92,
    });

    const cloud = new THREE.Points(geometry, material);
    cloud.frustumCulled = false; // drawRange를 쓰면 바운딩이 실제보다 커 보인다
    scene.add(cloud);
    cloudRef.current = cloud;
    totalRef.current = points.length;
    lastNotifiedRef.current = 0;
    countRef.current = MIN_POINTS;
    spinRef.current = true;
  }, [points]);

  return (
    <div className={`cloud-stage ${busy ? "is-busy" : ""}`}>
      <div className="cloud-canvas" ref={hostRef} />
      <span className="cloud-hint">Drag to rotate</span>
    </div>
  );
}
