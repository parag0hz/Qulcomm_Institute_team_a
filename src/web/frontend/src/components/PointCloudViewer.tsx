import { useEffect, useRef } from "react";
import * as THREE from "three";

/**
 * 모델이 실제로 보는 것을 그대로 화면에 띄운다.
 *
 * 차체 렌더링이 아니라 2,048개의 점 — 이게 PointNet의 입력 전부다. 관객이
 * "이 점들만 보고 항력을 맞힌다고?"라고 느끼는 순간이 이 데모의 핵심이라,
 * 매끄러운 메시로 미화하지 않고 점을 점으로 보여준다.
 */
interface PointCloudViewerProps {
  points: number[][] | null;
  busy?: boolean;
}

export function PointCloudViewer({ points, busy = false }: PointCloudViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const cloudRef = useRef<THREE.Points | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const spinRef = useRef(true);
  // 점군마다 크기·위치가 다르므로 카메라를 하드코딩하지 않고 bbox에서 계산한다.
  const frameRef = useRef({ cx: 1.5, cy: 0, cz: 0.6, radius: 7 });

  // 씬은 한 번만 만들고, 점군만 갈아 끼운다.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    camera.up.set(0, 0, 1);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    // 바닥 그리드 — 점군이 공중에 뜬 게 아니라 지면 위 차량임을 알려준다.
    const grid = new THREE.GridHelper(9, 18, 0x2a3a44, 0x18242b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const pivot = new THREE.Group();
    scene.add(pivot);

    let frame = 0;
    let dragging = false;
    let lastX = 0;
    let yaw = 0.6;

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
      const { clientWidth: w, clientHeight: h } = host;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const tick = () => {
      frame = requestAnimationFrame(tick);
      if (spinRef.current) yaw += 0.0032;
      const { cx, cy, cz, radius } = frameRef.current;
      camera.position.set(
        cx + Math.cos(yaw) * radius,
        cy + Math.sin(yaw) * radius,
        cz + radius * 0.34,
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

  // 점군 교체
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

    const material = new THREE.PointsMaterial({
      size: 0.05,
      color: 0x4fd8c8,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.95,
    });

    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    if (box) {
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      // 가장 긴 변이 화면에 여유 있게 들어오는 거리. 시야각 38도 기준.
      const span = Math.max(size.x, size.y, size.z);
      frameRef.current = {
        cx: center.x,
        cy: center.y,
        cz: center.z * 0.9,
        radius: span * 1.45,
      };
    }

    const cloud = new THREE.Points(geometry, material);
    scene.add(cloud);
    cloudRef.current = cloud;
    spinRef.current = true;
  }, [points]);

  return (
    <div className={`cloud-stage ${busy ? "is-busy" : ""}`}>
      <div className="cloud-canvas" ref={hostRef} />
      <span className="cloud-hint">드래그해서 돌려보세요 · 2,048 points</span>
    </div>
  );
}
