import { useEffect, useRef } from "react";
import * as THREE from "three";

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
  /** 밀도 애니메이션을 멈추고 이 개수로 고정한다(추론 결과를 볼 때). */
  freezeAt?: number | null;
  onDensityChange?: (count: number) => void;
}

const FOV = 34;
const MIN_POINTS = 96;
// 한 번 차오르고 빠지는 데 걸리는 시간. 너무 빠르면 산만하고 느리면 지루하다.
const CYCLE_MS = 7200;

export function PointCloudViewer({
  points,
  busy = false,
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
      const cloud = cloudRef.current;
      const total = totalRef.current;
      if (cloud && total > 0) {
        let count: number;
        if (freezeRef.current) {
          count = Math.min(freezeRef.current, total);
        } else {
          // 삼각파를 부드럽게 다듬어 차오르고 빠지는 호흡을 만든다.
          const phase = ((performance.now() - started) % CYCLE_MS) / CYCLE_MS;
          const triangle = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
          const eased = triangle * triangle * (3 - 2 * triangle);
          count = Math.round(MIN_POINTS + (total - MIN_POINTS) * eased);
        }
        cloud.geometry.setDrawRange(0, count);
        if (Math.abs(count - lastNotifiedRef.current) > total * 0.02) {
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
      cancelAnimationFrame(frame);
      observer.disconnect();
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
        // 3/4 각도에서 가로로 필요한 폭은 길이와 폭 사이 어딘가다.
        spanXY: Math.hypot(size.x, size.y) * 0.82,
        spanZ: size.z,
      };
    }

    const material = new THREE.PointsMaterial({
      size: 0.042,
      color: 0xdbe7ef,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.92,
    });

    const cloud = new THREE.Points(geometry, material);
    cloud.frustumCulled = false; // drawRange를 쓰면 바운딩이 실제보다 커 보인다
    scene.add(cloud);
    cloudRef.current = cloud;
    totalRef.current = points.length;
    lastNotifiedRef.current = 0;
    spinRef.current = true;
  }, [points]);

  return (
    <div className={`cloud-stage ${busy ? "is-busy" : ""}`}>
      <div className="cloud-canvas" ref={hostRef} />
      <span className="cloud-hint">Drag to rotate</span>
    </div>
  );
}
