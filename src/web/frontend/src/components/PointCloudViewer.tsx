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
  onDensityChange?: (count: number) => void;
}

const FOV = 34;
const MIN_POINTS = 96;
// 차체가 점으로 녹아내리는 시간. 짧으면 못 보고 길면 기다리게 된다.
const DISSOLVE_MS = 1900;
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
  const notifyRef = useRef<((count: number) => void) | undefined>(undefined);
  const lastNotifiedRef = useRef(0);
  // 고정 지점으로 뚝 끊지 않고 흘러들어가게 한다.
  const countRef = useRef(0);
  const meshRef = useRef<THREE.Group | null>(null);
  const meshMatsRef = useRef<THREE.MeshStandardMaterial[]>([]);
  // 인트로가 끝나기 전에는 밀도 호흡을 시작하지 않는다.
  const dissolveStartRef = useRef<number | null>(null);
  const introDoneRef = useRef(false);
  const introEndRef = useRef<number | null>(null);

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
          dissolveStartRef.current = performance.now();
        })
        .catch(() => {
          // 메시를 못 받아도 데모는 점군만으로 성립한다.
          introDoneRef.current = true;
        });
    } else {
      introDoneRef.current = true;
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
      // 인트로: 차체가 옅어지는 만큼 점이 차오른다. 두 동작이 같은 진행률을
      // 공유해야 '녹아내린다'로 읽히고, 따로 놀면 그냥 겹쳐 보인다.
      const cloud = cloudRef.current;
      const total = totalRef.current;
      const dissolveStart = dissolveStartRef.current;
      if (dissolveStart !== null && !introDoneRef.current) {
        const raw = Math.min(1, (performance.now() - dissolveStart) / DISSOLVE_MS);
        const progress = raw * raw * (3 - 2 * raw);
        meshMatsRef.current.forEach((material) => {
          material.opacity = 1 - progress;
        });
        if (meshRef.current) meshRef.current.visible = progress < 1;
        if (cloud) {
          (cloud.material as THREE.PointsMaterial).opacity = 0.92 * progress;
          if (total > 0) {
            countRef.current = MIN_POINTS + (total - MIN_POINTS) * progress;
            cloud.geometry.setDrawRange(0, Math.max(1, Math.round(countRef.current)));
          }
        }
        if (raw >= 1) {
          introDoneRef.current = true;
          // 호흡이 지금 밀도에서 자연스럽게 이어지도록 위상을 맞춘다.
          introEndRef.current = performance.now();
        }
      }

      if (cloud && total > 0 && introDoneRef.current) {
        if (freezeRef.current) {
          // 결과가 나오면 모델이 실제로 읽은 개수로 수렴시킨다. 순간이동시키면
          // 화면이 갑자기 성겨져 고장처럼 보이므로 흘러들어가게 둔다.
          const target = Math.min(freezeRef.current, total);
          countRef.current += (target - countRef.current) * 0.07;
          if (Math.abs(target - countRef.current) < 2) countRef.current = target;
        } else {
          // 삼각파를 부드럽게 다듬어 차오르고 빠지는 호흡을 만든다.
          // 인트로가 끝난 지점(밀도 최대)에서 이어받도록 반주기 오프셋을 준다.
          const base = introEndRef.current ?? started;
          const phase = (((performance.now() - base) + CYCLE_MS / 2) % CYCLE_MS) / CYCLE_MS;
          const triangle = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
          const eased = triangle * triangle * (3 - 2 * triangle);
          countRef.current = MIN_POINTS + (total - MIN_POINTS) * eased;
        }
        const count = Math.max(1, Math.round(countRef.current));
        cloud.geometry.setDrawRange(0, count);
        // 호흡 중에는 리렌더를 아끼려 큰 변화만 알린다. 다만 고정 지점에
        // 도달했을 때는 반드시 정확한 값을 알려야 한다 — 화면이 2,048점을
        // 그리면서 라벨이 2,076이라고 말하면 그건 거짓말이다.
        const settled =
          freezeRef.current != null && count === Math.min(freezeRef.current, total);
        const changedEnough = Math.abs(count - lastNotifiedRef.current) > total * 0.02;
        if (settled ? lastNotifiedRef.current !== count : changedEnough) {
          lastNotifiedRef.current = count;
          notifyRef.current?.(count);
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
      opacity: introDoneRef.current ? 0.92 : 0,
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
