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
}

const FOV = 34;

export function PointCloudViewer({ points, busy = false }: PointCloudViewerProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const cloudRef = useRef<THREE.Points | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const spinRef = useRef(true);
  // Framing is derived from the cloud itself so every body type fills the frame.
  const fitRef = useRef({ cx: 1.5, cy: 0, cz: 0.6, radius: 3.2 });

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

    const tick = () => {
      frame = requestAnimationFrame(tick);
      if (spinRef.current) yaw += 0.0026;

      const { cx, cy, cz, radius } = fitRef.current;
      // Fit the bounding sphere in whichever axis is tighter, so a wide panel
      // and a narrow one both show the whole car.
      const vFov = (FOV * Math.PI) / 180;
      const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
      const distance = (radius / Math.sin(Math.min(vFov, hFov) / 2)) * 1.12;

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
    geometry.computeBoundingSphere();

    const sphere = geometry.boundingSphere;
    if (sphere) {
      fitRef.current = {
        cx: sphere.center.x,
        cy: sphere.center.y,
        cz: sphere.center.z * 0.85,
        radius: sphere.radius,
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
    scene.add(cloud);
    cloudRef.current = cloud;
    spinRef.current = true;
  }, [points]);

  return (
    <div className={`cloud-stage ${busy ? "is-busy" : ""}`}>
      <div className="cloud-canvas" ref={hostRef} />
      <span className="cloud-hint">
        Drag to rotate · {(points?.length ?? 0).toLocaleString()} points
      </span>
    </div>
  );
}
