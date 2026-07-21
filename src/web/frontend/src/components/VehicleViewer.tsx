import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type {
  NumericDesignValues,
  NumericParameterName,
  ParameterDefinition,
  WheelTreatment,
} from "../types";

export type CameraView = "perspective" | "side" | "front" | "top";

export interface VehicleViewerStatus {
  referenceKind: "glb" | "fallback";
  message: string;
  datasetWheelCount: number;
}

export interface VehicleViewerHandle {
  setView: (view: CameraView) => void;
  resetView: () => void;
  captureThumbnail: (quality?: number) => string;
  showReference: () => void;
}

export interface VehicleViewerProps {
  /** Numeric design values. Kept as object so the full 25-feature design can be passed safely. */
  values: Readonly<NumericDesignValues>;
  parameters: readonly ParameterDefinition[];
  wheelTreatment?: WheelTreatment;
  activeParameter?: NumericParameterName | null;
  dimensionsVisible?: boolean;
  importedPoints?: readonly (readonly number[])[] | null;
  modelUrl?: string;
  className?: string;
  onLoadingChange?: (loading: boolean) => void;
  onStatusChange?: (status: VehicleViewerStatus) => void;
  onError?: (message: string) => void;
}

type ReferenceKind = "glb" | "fallback";
type DimensionKey = "length" | "width" | "height" | "local";

interface MorphEntry {
  mesh: THREE.Mesh;
  original: Float32Array;
}

interface WheelSpec {
  name: string;
  center: THREE.Vector3;
  radius: number;
  width: number;
}

interface DimensionEntry {
  line: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  point: THREE.Vector3;
}

interface DimensionsState extends Record<DimensionKey, DimensionEntry> {
  group: THREE.Group;
}

interface ViewerRuntime {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  referenceModel: THREE.Group | null;
  referenceBounds: THREE.Box3 | null;
  referenceKind: ReferenceKind;
  morphMeshes: MorphEntry[];
  datasetWheels: THREE.Mesh[];
  proceduralWheels: Map<string, THREE.Group>;
  wheelSpecs: WheelSpec[];
  dimensions: DimensionsState;
  importedCloud: THREE.Points | null;
  resizeObserver: ResizeObserver | null;
  removeFallbackResize: (() => void) | null;
  destroyed: boolean;
}

interface MorphRule {
  kind: string;
  dimension?: Exclude<DimensionKey, "local">;
  anchor?: readonly [number, number, number];
}

const MODEL_URL = "/static/models/drivaer_reference.glb";
const VIEW_POSITIONS: Record<CameraView, readonly [number, number, number]> = {
  perspective: [-6.2, -6.4, 3.4],
  side: [0, -8.5, 1.45],
  front: [-8.5, 0, 1.4],
  top: [0, 0, 9.5],
};

const MORPH_RULES: Record<string, MorphRule> = {
  A_Car_Length: { kind: "length", dimension: "length" },
  A_Car_Width: { kind: "width", dimension: "width" },
  A_Car_Roof_Height: { kind: "roof", dimension: "height" },
  A_Car_Green_House_Angle: { kind: "greenhouse", anchor: [0.5, -0.88, 0.82] },
  B_Ramp_Angle: { kind: "ramp", anchor: [0.84, -0.82, 0.25] },
  B_Diffusor_Angle: { kind: "diffuser", anchor: [0.93, -0.72, 0.12] },
  B_Trunklid_Angle: { kind: "trunkAngle", anchor: [0.84, -0.86, 0.66] },
  C_Side_Mirrors_Rotation: { kind: "marker", anchor: [0.38, -1.02, 0.68] },
  C_Side_Mirrors_Translate_X: { kind: "mirrorX", anchor: [0.38, -1.02, 0.68] },
  C_Side_Mirrors_Translate_Z: { kind: "mirrorZ", anchor: [0.38, -1.02, 0.68] },
  D_Rear_Window_Inclination: { kind: "rearWindowAngle", anchor: [0.7, -0.82, 0.82] },
  D_Winscreen_Inclination: { kind: "windscreenAngle", anchor: [0.34, -0.82, 0.82] },
  D_Winscreen_Length: { kind: "windscreenLength", anchor: [0.34, -0.82, 0.82] },
  D_Rear_Window_Length: { kind: "rearWindowLength", anchor: [0.7, -0.82, 0.82] },
  E_A_B_C_Pillar_Thickness: { kind: "marker", anchor: [0.58, -0.84, 0.82] },
  E_Fenders_Arch_Offset: { kind: "fender", anchor: [0.2, -0.98, 0.34] },
  F_Door_Handles_Thickness: { kind: "marker", anchor: [0.56, -1.01, 0.58] },
  F_Door_Handles_X_Position: { kind: "marker", anchor: [0.56, -1.01, 0.58] },
  F_Door_Handles_Z_Position: { kind: "marker", anchor: [0.56, -1.01, 0.58] },
  G_Trunklid_Curvature: { kind: "trunkCurve", anchor: [0.86, -0.86, 0.62] },
  G_Trunklid_Length: { kind: "trunkLength", anchor: [0.86, -0.86, 0.62] },
  H_Front_Bumper_Curvature: { kind: "bumperCurve", anchor: [0.03, -0.82, 0.35] },
  H_Front_Bumper_Length: { kind: "bumperLength", anchor: [0.03, -0.82, 0.35] },
};

const DIMENSION_KEYS: readonly DimensionKey[] = ["length", "width", "height", "local"];

function valueFrom(values: Readonly<object>, name: string, fallback = 0): number {
  const value = (values as Record<string, unknown>)[name];
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function compact(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(1).replace(/\.0$/, "");
  if (Math.abs(value) >= 10) return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${compact(value)}`;
}

function smoothMask(edge0: number, edge1: number, value: number): number {
  return THREE.MathUtils.smoothstep(value, edge0, edge1);
}

function bellMask(value: number, center: number, radius: number): number {
  return 1 - THREE.MathUtils.smoothstep(Math.abs(value - center), radius * 0.35, radius);
}

function setCameraView(runtime: ViewerRuntime, view: CameraView): void {
  runtime.camera.position.fromArray(VIEW_POSITIONS[view]);
  runtime.camera.up.set(0, view === "top" ? 1 : 0, view === "top" ? 0 : 1);
  runtime.controls.target.set(0, 0, 0.65);
  runtime.controls.update();
}

function createScene(canvas: HTMLCanvasElement, host: HTMLDivElement): ViewerRuntime {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xe7edef, 12, 24);
  const camera = new THREE.PerspectiveCamera(36, 1, 0.05, 100);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.target.set(0, 0, 0.65);
  controls.minDistance = 4;
  controls.maxDistance = 18;
  controls.maxPolarAngle = Math.PI * 0.49;

  scene.add(new THREE.HemisphereLight(0xf8ffff, 0x66747a, 2.2));
  const keyLight = new THREE.DirectionalLight(0xffffff, 4.2);
  keyLight.position.set(-4, -5, 8);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.left = -6;
  keyLight.shadow.camera.right = 6;
  keyLight.shadow.camera.top = 6;
  keyLight.shadow.camera.bottom = -6;
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x78d6ca, 2);
  rimLight.position.set(5, 4, 3);
  scene.add(rimLight);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(24, 24),
    new THREE.MeshStandardMaterial({ color: 0xd9e1e3, roughness: 0.95, metalness: 0 }),
  );
  ground.position.z = -0.018;
  ground.receiveShadow = true;
  scene.add(ground);
  const grid = new THREE.GridHelper(20, 40, 0xaebbc0, 0xcbd4d7);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.01;
  const gridMaterial = grid.material as THREE.Material;
  gridMaterial.opacity = 0.32;
  gridMaterial.transparent = true;
  scene.add(grid);

  const dimensions = createDimensions(scene);
  const runtime: ViewerRuntime = {
    renderer,
    scene,
    camera,
    controls,
    referenceModel: null,
    referenceBounds: null,
    referenceKind: "glb",
    morphMeshes: [],
    datasetWheels: [],
    proceduralWheels: new Map(),
    wheelSpecs: [],
    dimensions,
    importedCloud: null,
    resizeObserver: null,
    removeFallbackResize: null,
    destroyed: false,
  };
  setCameraView(runtime, "perspective");

  const resize = () => {
    const rect = host.getBoundingClientRect();
    const width = Math.max(1, rect.width || host.clientWidth);
    const height = Math.max(1, rect.height || host.clientHeight);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  resize();
  if (typeof ResizeObserver !== "undefined") {
    runtime.resizeObserver = new ResizeObserver(resize);
    runtime.resizeObserver.observe(host);
  } else {
    window.addEventListener("resize", resize);
    runtime.removeFallbackResize = () => window.removeEventListener("resize", resize);
  }
  return runtime;
}

function createDimensions(scene: THREE.Scene): DimensionsState {
  const group = new THREE.Group();
  group.name = "Paragon_Dimensions";
  const createEntry = (local: boolean): DimensionEntry => {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({
      color: local ? 0x087f78 : 0x52747a,
      transparent: true,
      opacity: local ? 1 : 0.66,
      depthTest: false,
    });
    const line = new THREE.LineSegments(geometry, material);
    line.renderOrder = 20;
    group.add(line);
    return { line, point: new THREE.Vector3() };
  };
  const dimensions = {
    group,
    length: createEntry(false),
    width: createEntry(false),
    height: createEntry(false),
    local: createEntry(true),
  };
  scene.add(group);
  return dimensions;
}

function createFallbackVehicle(): THREE.Group {
  const group = new THREE.Group();
  group.name = "Fallback_DrivAer";
  const bodyMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x087f78,
    roughness: 0.3,
    metalness: 0.35,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(4.6, 1.85, 0.55), bodyMaterial);
  body.name = "Body";
  body.position.z = 0.55;
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.5, 1.58, 0.65), bodyMaterial);
  cabin.name = "Cabin";
  cabin.position.set(0.35, 0, 1.1);
  group.add(body, cabin);
  const wheelMaterial = new THREE.MeshStandardMaterial({ color: 0x1b2227, roughness: 0.8 });
  const positions: Array<[string, number, number]> = [
    ["Wheel_FL", -1.45, -0.95],
    ["Wheel_FR", -1.45, 0.95],
    ["Wheel_RL", 1.45, -0.95],
    ["Wheel_RR", 1.45, 0.95],
  ];
  for (const [name, x, y] of positions) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.34, 0.18, 24), wheelMaterial);
    wheel.name = name;
    wheel.rotation.x = Math.PI / 2;
    wheel.position.set(x, y, 0.34);
    group.add(wheel);
  }
  group.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  return group;
}

function createProceduralWheel(spec: WheelSpec, treatment: string): THREE.Group {
  const assembly = new THREE.Group();
  assembly.name = `${spec.name}_${treatment.replaceAll(" ", "_")}`;
  assembly.position.copy(spec.center);
  assembly.userData.baseCenter = spec.center.clone();
  const tireMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x111518,
    roughness: 0.72,
    metalness: 0.03,
    clearcoat: 0.12,
  });
  const rimMaterial = new THREE.MeshPhysicalMaterial({
    color: treatment === "Closed smooth" ? 0x9ba7aa : 0xaeb8ba,
    roughness: 0.28,
    metalness: 0.72,
    clearcoat: 0.35,
  });
  const tire = new THREE.Mesh(
    new THREE.TorusGeometry(spec.radius * 0.78, spec.radius * 0.22, 18, 64),
    tireMaterial,
  );
  tire.rotation.x = Math.PI / 2;
  assembly.add(tire);
  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(spec.radius * 0.57, spec.radius * 0.065, 12, 48),
    rimMaterial,
  );
  rim.rotation.x = Math.PI / 2;
  assembly.add(rim);
  const hub = new THREE.Mesh(
    new THREE.CylinderGeometry(spec.radius * 0.13, spec.radius * 0.13, spec.width * 0.78, 32),
    rimMaterial,
  );
  assembly.add(hub);
  if (treatment === "Closed smooth") {
    assembly.add(new THREE.Mesh(
      new THREE.CylinderGeometry(spec.radius * 0.7, spec.radius * 0.7, spec.width * 0.42, 64),
      rimMaterial,
    ));
  } else {
    for (let index = 0; index < 5; index += 1) {
      const angle = index * Math.PI * 2 / 5;
      const spoke = new THREE.Mesh(
        new THREE.BoxGeometry(spec.radius * 0.48, spec.width * 0.34, spec.radius * 0.105),
        rimMaterial,
      );
      spoke.position.set(Math.cos(angle) * spec.radius * 0.36, 0, Math.sin(angle) * spec.radius * 0.36);
      spoke.rotation.y = -angle;
      assembly.add(spoke);
    }
  }
  assembly.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  return assembly;
}

function configureReference(runtime: ViewerRuntime, model: THREE.Group, kind: ReferenceKind): void {
  const bounds = new THREE.Box3().setFromObject(model);
  const center = bounds.getCenter(new THREE.Vector3());
  model.position.set(-center.x, -center.y, -bounds.min.z);
  runtime.referenceKind = kind;
  runtime.referenceBounds = bounds.clone();
  runtime.referenceModel = model;
  runtime.morphMeshes = [];
  runtime.datasetWheels = [];
  runtime.wheelSpecs = [];
  runtime.proceduralWheels.clear();

  model.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.castShadow = true;
    object.receiveShadow = true;
    if (object.name === "Body") {
      const position = object.geometry.getAttribute("position");
      if (kind === "glb" && position) {
        runtime.morphMeshes.push({
          mesh: object,
          original: Float32Array.from(position.array as ArrayLike<number>),
        });
      }
      object.material = new THREE.MeshPhysicalMaterial({
        color: 0x087f78,
        metalness: 0.46,
        roughness: 0.24,
        clearcoat: 0.72,
        clearcoatRoughness: 0.2,
      });
    } else if (object.name.startsWith("Wheel_")) {
      object.geometry.computeBoundingBox();
      const wheelBounds = object.geometry.boundingBox?.clone() ?? new THREE.Box3();
      const geometryCenter = wheelBounds.getCenter(new THREE.Vector3());
      const wheelCenter = kind === "glb"
        ? geometryCenter
        : object.position.clone();
      const radius = Math.max(0.05, wheelBounds.max.z - geometryCenter.z);
      const width = Math.max(0.04, wheelBounds.max.y - wheelBounds.min.y);
      runtime.datasetWheels.push(object);
      runtime.wheelSpecs.push({ name: object.name, center: wheelCenter.clone(), radius, width });
      object.userData.baseCenter = wheelCenter.clone();
      object.material = new THREE.MeshPhysicalMaterial({
        color: 0x151b1f,
        metalness: 0.22,
        roughness: 0.52,
        clearcoat: 0.18,
      });
    }
  });

  if (kind === "glb" && (runtime.morphMeshes.length !== 1 || runtime.datasetWheels.length !== 4)) {
    throw new Error("Reference GLB must contain Body and four named Wheel nodes.");
  }
  for (const treatment of ["Open smooth", "Closed smooth"]) {
    const set = new THREE.Group();
    set.name = `Procedural_${treatment.replaceAll(" ", "_")}`;
    for (const spec of runtime.wheelSpecs) set.add(createProceduralWheel(spec, treatment));
    set.visible = false;
    model.add(set);
    runtime.proceduralWheels.set(treatment, set);
  }
  runtime.scene.add(model);
}

async function loadReference(runtime: ViewerRuntime, modelUrl: string): Promise<ReferenceKind> {
  try {
    const gltf = await new GLTFLoader().loadAsync(modelUrl);
    if (runtime.destroyed) {
      disposeObject(gltf.scene);
      return "glb";
    }
    configureReference(runtime, gltf.scene, "glb");
    return "glb";
  } catch {
    if (runtime.destroyed) return "fallback";
    if (runtime.referenceModel) {
      runtime.scene.remove(runtime.referenceModel);
      disposeObject(runtime.referenceModel);
      runtime.referenceModel = null;
    }
    const fallback = createFallbackVehicle();
    configureReference(runtime, fallback, "fallback");
    return "fallback";
  }
}

function normalizedParameters(
  values: Readonly<object>,
  parameters: readonly ParameterDefinition[],
): Record<string, number> {
  return Object.fromEntries(parameters.map((parameter) => {
    const value = valueFrom(values, parameter.name, parameter.default);
    const span = value >= parameter.default
      ? parameter.max - parameter.default
      : parameter.default - parameter.min;
    return [
      parameter.name,
      THREE.MathUtils.clamp((value - parameter.default) / Math.max(span, 1e-9), -1, 1),
    ];
  }));
}

function updateWheelSelection(runtime: ViewerRuntime, treatment: string): void {
  for (const wheel of runtime.datasetWheels) wheel.visible = treatment === "Open detailed";
  for (const [name, group] of runtime.proceduralWheels) group.visible = treatment === name;
}

function updateWheelPositions(runtime: ViewerRuntime, normalized: Record<string, number>): void {
  if (!runtime.referenceBounds) return;
  const center = runtime.referenceBounds.getCenter(new THREE.Vector3());
  const lengthScale = 1 + 0.05 * (normalized.A_Car_Length ?? 0);
  const widthScale = 1 + 0.04 * (normalized.A_Car_Width ?? 0);
  const targetFor = (base: THREE.Vector3) => new THREE.Vector3(
    center.x + (base.x - center.x) * lengthScale,
    center.y + (base.y - center.y) * widthScale,
    base.z,
  );
  for (const wheel of runtime.datasetWheels) {
    const base = wheel.userData.baseCenter as THREE.Vector3;
    wheel.position.copy(targetFor(base)).sub(base);
  }
  for (const group of runtime.proceduralWheels.values()) {
    for (const assembly of group.children) {
      const base = assembly.userData.baseCenter as THREE.Vector3;
      assembly.position.copy(targetFor(base));
    }
  }
}

function applyGeometryMorph(
  runtime: ViewerRuntime,
  values: Readonly<object>,
  parameters: readonly ParameterDefinition[],
): void {
  if (!runtime.referenceModel || !runtime.referenceBounds || !parameters.length) return;
  const n = normalizedParameters(values, parameters);
  if (runtime.referenceKind === "fallback") {
    runtime.referenceModel.scale.set(
      1 + 0.05 * (n.A_Car_Length ?? 0),
      1 + 0.04 * (n.A_Car_Width ?? 0),
      1 + 0.06 * (n.A_Car_Roof_Height ?? 0),
    );
    return;
  }
  const base = runtime.referenceBounds;
  const size = base.getSize(new THREE.Vector3());
  const center = base.getCenter(new THREE.Vector3());
  for (const entry of runtime.morphMeshes) {
    const position = entry.mesh.geometry.getAttribute("position") as THREE.BufferAttribute;
    const target = position.array as Float32Array;
    const source = entry.original;
    for (let index = 0; index < source.length; index += 3) {
      const ox = source[index];
      const oy = source[index + 1];
      const oz = source[index + 2];
      const ux = (ox - base.min.x) / size.x;
      const uy = Math.abs((oy - center.y) / (size.y * 0.5));
      const uz = (oz - base.min.z) / size.z;
      let x = center.x + (ox - center.x) * (1 + 0.05 * (n.A_Car_Length ?? 0));
      let y = center.y + (oy - center.y) * (1 + 0.04 * (n.A_Car_Width ?? 0));
      let z = oz;

      const upper = smoothMask(0.42, 0.78, uz);
      const greenhouse = smoothMask(0.52, 0.78, uz);
      z += size.z * 0.06 * (n.A_Car_Roof_Height ?? 0) * upper;
      y = center.y + (y - center.y) * (1 - 0.035 * (n.A_Car_Green_House_Angle ?? 0) * greenhouse);
      x += size.x * 0.018 * (n.A_Car_Green_House_Angle ?? 0) * greenhouse * (ux - 0.5);

      const rear = smoothMask(0.68, 0.98, ux);
      const front = 1 - smoothMask(0.03, 0.26, ux);
      const bottom = 1 - smoothMask(0.12, 0.34, uz);
      const topRear = rear * smoothMask(0.42, 0.7, uz);
      z += size.z * 0.045 * (n.B_Diffusor_Angle ?? 0) * rear * bottom;
      z += size.z * 0.025 * (n.B_Ramp_Angle ?? 0) * smoothMask(0.56, 0.9, ux) * bottom;
      z += size.z * 0.035 * (n.B_Trunklid_Angle ?? 0) * topRear * (ux - 0.68) / 0.32;
      x += size.x * 0.025 * (n.G_Trunklid_Length ?? 0) * topRear;
      z += size.z * 0.022 * (n.G_Trunklid_Curvature ?? 0) * topRear * bellMask(ux, 0.84, 0.18);
      x -= size.x * 0.022 * (n.H_Front_Bumper_Length ?? 0) * front * (1 - 0.25 * uz);
      x -= size.x * 0.012 * (n.H_Front_Bumper_Curvature ?? 0) * front * bellMask(uz, 0.35, 0.38);

      const windscreen = bellMask(ux, 0.34, 0.16) * greenhouse;
      const rearWindow = bellMask(ux, 0.69, 0.17) * greenhouse;
      x += size.x * 0.025 * (n.D_Winscreen_Inclination ?? 0) * windscreen * (uz - 0.58);
      x -= size.x * 0.025 * (n.D_Rear_Window_Inclination ?? 0) * rearWindow * (uz - 0.58);
      x += size.x * 0.016 * (n.D_Winscreen_Length ?? 0) * windscreen;
      x -= size.x * 0.016 * (n.D_Rear_Window_Length ?? 0) * rearWindow;

      const mirror = bellMask(ux, 0.38, 0.12)
        * smoothMask(0.82, 0.98, uy)
        * bellMask(uz, 0.66, 0.2);
      x += size.x * 0.018 * (n.C_Side_Mirrors_Translate_X ?? 0) * mirror;
      z += size.z * 0.035 * (n.C_Side_Mirrors_Translate_Z ?? 0) * mirror;
      const wheelArch = (bellMask(ux, 0.2, 0.14) + bellMask(ux, 0.78, 0.14))
        * smoothMask(0.72, 0.96, uy)
        * bellMask(uz, 0.28, 0.25);
      y += Math.sign(oy - center.y) * size.y * 0.025
        * (n.E_Fenders_Arch_Offset ?? 0) * Math.min(1, wheelArch);

      target[index] = x;
      target[index + 1] = y;
      target[index + 2] = Math.max(base.min.z, z);
    }
    position.needsUpdate = true;
    entry.mesh.geometry.computeVertexNormals();
    entry.mesh.geometry.computeBoundingBox();
    entry.mesh.geometry.computeBoundingSphere();
  }
  updateWheelPositions(runtime, n);
}

function dimensionSegments(
  start: THREE.Vector3,
  end: THREE.Vector3,
  tickVector: THREE.Vector3,
): THREE.Vector3[] {
  return [
    start,
    end,
    start.clone().sub(tickVector),
    start.clone().add(tickVector),
    end.clone().sub(tickVector),
    end.clone().add(tickVector),
  ];
}

function setDimensionLine(entry: DimensionEntry, points: THREE.Vector3[]): void {
  entry.line.geometry.setFromPoints(points);
  entry.line.geometry.computeBoundingSphere();
  entry.point.copy(points[0]).add(points[1]).multiplyScalar(0.5);
}

function updateDimensions(
  runtime: ViewerRuntime,
  values: Readonly<object>,
  parameters: readonly ParameterDefinition[],
  activeParameter: NumericParameterName | null | undefined,
  visible: boolean,
  labels: Record<DimensionKey, HTMLDivElement | null>,
): void {
  if (!runtime.referenceModel) return;
  runtime.dimensions.group.visible = visible;
  if (!visible) return;
  const bounds = new THREE.Box3().setFromObject(runtime.referenceModel);
  const size = bounds.getSize(new THREE.Vector3());
  const tick = Math.max(size.x, size.y) * 0.025;
  const rule: MorphRule = activeParameter
    ? MORPH_RULES[activeParameter] ?? { kind: "none" }
    : { kind: "none" };
  const xZ = bounds.min.z + size.z * 0.08;
  setDimensionLine(runtime.dimensions.length, dimensionSegments(
    new THREE.Vector3(bounds.min.x, bounds.min.y - size.y * 0.13, xZ),
    new THREE.Vector3(bounds.max.x, bounds.min.y - size.y * 0.13, xZ),
    new THREE.Vector3(0, 0, tick),
  ));
  setDimensionLine(runtime.dimensions.width, dimensionSegments(
    new THREE.Vector3(bounds.min.x + size.x * 0.08, bounds.min.y, xZ),
    new THREE.Vector3(bounds.min.x + size.x * 0.08, bounds.max.y, xZ),
    new THREE.Vector3(tick, 0, 0),
  ));
  setDimensionLine(runtime.dimensions.height, dimensionSegments(
    new THREE.Vector3(bounds.max.x + size.x * 0.06, bounds.max.y + size.y * 0.08, bounds.min.z),
    new THREE.Vector3(bounds.max.x + size.x * 0.06, bounds.max.y + size.y * 0.08, bounds.max.z),
    new THREE.Vector3(tick, 0, 0),
  ));

  const copy: Record<Exclude<DimensionKey, "local">, string> = {
    length: `Length Δ ${signed(valueFrom(values, "A_Car_Length") - (parameters.find((item) => item.name === "A_Car_Length")?.default ?? 0))} mm`,
    width: `Width Δ ${signed(valueFrom(values, "A_Car_Width") - (parameters.find((item) => item.name === "A_Car_Width")?.default ?? 0))} mm`,
    height: `Roof Δ ${signed(valueFrom(values, "A_Car_Roof_Height") - (parameters.find((item) => item.name === "A_Car_Roof_Height")?.default ?? 0))} mm`,
  };
  for (const key of ["length", "width", "height"] as const) {
    const active = rule.dimension === key;
    const label = labels[key];
    if (label) {
      label.textContent = copy[key];
      label.classList.toggle("active", active);
    }
    runtime.dimensions[key].line.material.color.set(active ? 0x087f78 : 0x52747a);
    runtime.dimensions[key].line.material.opacity = active ? 1 : 0.6;
  }

  const local = runtime.dimensions.local;
  const parameter = parameters.find((item) => item.name === activeParameter);
  const localLabel = labels.local;
  if (parameter && rule.anchor && !rule.dimension && runtime.referenceKind !== "fallback") {
    const anchor = new THREE.Vector3(
      THREE.MathUtils.lerp(bounds.min.x, bounds.max.x, rule.anchor[0]),
      THREE.MathUtils.lerp(bounds.min.y, bounds.max.y, (rule.anchor[1] + 1) * 0.5),
      THREE.MathUtils.lerp(bounds.min.z, bounds.max.z, rule.anchor[2]),
    );
    const end = anchor.clone().add(new THREE.Vector3(size.x * 0.11, -size.y * 0.09, size.z * 0.12));
    setDimensionLine(local, [anchor, end]);
    local.point.copy(end);
    local.line.visible = true;
    if (localLabel) {
      const isAngle = ["Angle", "Inclination", "Rotation"].some((token) => parameter.name.includes(token));
      const value = valueFrom(values, parameter.name, parameter.default);
      localLabel.textContent = `${parameter.label}  ${isAngle ? `${compact(value)}°` : `Δ ${signed(value - parameter.default)} mm`}`;
      localLabel.classList.remove("hidden");
    }
  } else {
    local.line.visible = false;
    localLabel?.classList.add("hidden");
  }
}

function updateDimensionLabels(
  runtime: ViewerRuntime,
  canvas: HTMLCanvasElement,
  labels: Record<DimensionKey, HTMLDivElement | null>,
  visible: boolean,
): void {
  if (!visible) return;
  const rect = canvas.getBoundingClientRect();
  for (const key of DIMENSION_KEYS) {
    const entry = runtime.dimensions[key];
    const label = labels[key];
    if (!label || !entry.line.visible) continue;
    const projected = entry.point.clone().project(runtime.camera);
    label.style.left = `${(projected.x * 0.5 + 0.5) * rect.width}px`;
    label.style.top = `${(-projected.y * 0.5 + 0.5) * rect.height}px`;
    label.classList.toggle("offscreen", projected.z < -1 || projected.z > 1);
  }
}

function setImportedPointCloud(runtime: ViewerRuntime, points: readonly (readonly number[])[] | null): void {
  if (runtime.importedCloud) {
    runtime.scene.remove(runtime.importedCloud);
    runtime.importedCloud.geometry.dispose();
    (runtime.importedCloud.material as THREE.Material).dispose();
    runtime.importedCloud = null;
  }
  if (!points?.length) {
    if (runtime.referenceModel) runtime.referenceModel.visible = true;
    return;
  }
  const positions = new Float32Array(points.length * 3);
  points.forEach((point, index) => {
    positions[index * 3] = Number(point[0] ?? 0);
    positions[index * 3 + 1] = Number(point[1] ?? 0);
    positions[index * 3 + 2] = Number(point[2] ?? 0);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.center();
  const material = new THREE.PointsMaterial({ color: 0x66d4c2, size: 0.025, sizeAttenuation: true });
  runtime.importedCloud = new THREE.Points(geometry, material);
  runtime.scene.add(runtime.importedCloud);
  if (runtime.referenceModel) runtime.referenceModel.visible = false;
}

function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.LineSegments)) return;
    const geometry = object.geometry as THREE.BufferGeometry | undefined;
    geometry?.dispose();
    const material = object.material as THREE.Material | THREE.Material[] | undefined;
    if (Array.isArray(material)) material.forEach((item) => item.dispose());
    else material?.dispose();
  });
}

function destroyRuntime(runtime: ViewerRuntime): void {
  runtime.destroyed = true;
  runtime.renderer.setAnimationLoop(null);
  runtime.resizeObserver?.disconnect();
  runtime.removeFallbackResize?.();
  runtime.controls.dispose();
  disposeObject(runtime.scene);
  runtime.scene.clear();
  runtime.renderer.dispose();
}

export const VehicleViewer = forwardRef<VehicleViewerHandle, VehicleViewerProps>(function VehicleViewer(
  {
    values,
    parameters,
    wheelTreatment = "Open detailed",
    activeParameter = null,
    dimensionsVisible = true,
    importedPoints = null,
    modelUrl = MODEL_URL,
    className = "",
    onLoadingChange,
    onStatusChange,
    onError,
  },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const runtimeRef = useRef<ViewerRuntime | null>(null);
  const latestRef = useRef({ values, parameters, wheelTreatment, activeParameter, dimensionsVisible, importedPoints });
  const labelsRef = useRef<Record<DimensionKey, HTMLDivElement | null>>({
    length: null,
    width: null,
    height: null,
    local: null,
  });
  const [loading, setLoading] = useState(true);
  const [fallbackMessage, setFallbackMessage] = useState("");
  latestRef.current = { values, parameters, wheelTreatment, activeParameter, dimensionsVisible, importedPoints };

  useImperativeHandle(ref, () => ({
    setView(view) {
      const runtime = runtimeRef.current;
      if (runtime) setCameraView(runtime, view);
    },
    resetView() {
      const runtime = runtimeRef.current;
      if (runtime) setCameraView(runtime, "perspective");
    },
    captureThumbnail(quality = 0.58) {
      const runtime = runtimeRef.current;
      const canvas = canvasRef.current;
      if (!runtime || !canvas) return "";
      try {
        runtime.renderer.render(runtime.scene, runtime.camera);
        return canvas.toDataURL("image/jpeg", THREE.MathUtils.clamp(quality, 0.1, 1));
      } catch {
        return "";
      }
    },
    showReference() {
      const runtime = runtimeRef.current;
      if (!runtime) return;
      setImportedPointCloud(runtime, null);
    },
  }), []);

  useLayoutEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    if (!host || !canvas) return;
    let runtime: ViewerRuntime;
    try {
      runtime = createScene(canvas, host);
    } catch {
      const message = "3D preview is unavailable because WebGL could not be initialized. Enable browser hardware acceleration or use a WebGL 2-capable browser.";
      setLoading(false);
      setFallbackMessage(message);
      onLoadingChange?.(false);
      onError?.(message);
      return;
    }
    runtimeRef.current = runtime;
    const startLoading = () => {
      setLoading(true);
      onLoadingChange?.(true);
    };
    const finishLoading = () => {
      setLoading(false);
      onLoadingChange?.(false);
    };
    startLoading();
    void loadReference(runtime, modelUrl)
      .then((kind) => {
        if (runtime.destroyed) return;
        const latest = latestRef.current;
        applyGeometryMorph(runtime, latest.values, latest.parameters);
        updateWheelSelection(runtime, latest.wheelTreatment);
        setImportedPointCloud(runtime, latest.importedPoints);
        updateDimensions(
          runtime,
          latest.values,
          latest.parameters,
          latest.activeParameter,
          latest.dimensionsVisible,
          labelsRef.current,
        );
        if (kind === "fallback") {
          const message = "Reference GLB could not be loaded; showing a simplified fallback vehicle.";
          setFallbackMessage(message);
          onError?.(message);
        }
        onStatusChange?.({
          referenceKind: kind,
          message: kind === "glb"
            ? "QEM body · 112k faces · dataset/procedural wheels"
            : "Simplified fallback preview",
          datasetWheelCount: runtime.datasetWheels.length,
        });
      })
      .catch((error: unknown) => {
        if (runtime.destroyed) return;
        const message = error instanceof Error ? error.message : "Reference geometry could not be initialized.";
        setFallbackMessage(message);
        onError?.(message);
      })
      .finally(() => {
        if (!runtime.destroyed) finishLoading();
      });

    runtime.renderer.setAnimationLoop(() => {
      runtime.controls.update();
      updateDimensionLabels(
        runtime,
        canvas,
        labelsRef.current,
        latestRef.current.dimensionsVisible,
      );
      runtime.renderer.render(runtime.scene, runtime.camera);
    });
    return () => {
      destroyRuntime(runtime);
      runtimeRef.current = null;
    };
  }, [modelUrl]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime?.referenceModel) return;
    applyGeometryMorph(runtime, values, parameters);
    updateDimensions(runtime, values, parameters, activeParameter, dimensionsVisible, labelsRef.current);
  }, [values, parameters, activeParameter, dimensionsVisible]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime?.referenceModel) return;
    updateWheelSelection(runtime, wheelTreatment);
  }, [wheelTreatment]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    if (!runtime) return;
    setImportedPointCloud(runtime, importedPoints);
  }, [importedPoints]);

  return (
    <div className={`viewer-host ${className}`.trim()} ref={hostRef} data-testid="vehicle-viewer">
      <canvas ref={canvasRef} aria-label="Interactive DrivAer reference vehicle" />
      <div
        className={`dimension-overlay${dimensionsVisible ? "" : " hidden"}`}
        aria-live="polite"
        aria-hidden={!dimensionsVisible}
      >
        {DIMENSION_KEYS.map((key) => (
          <div
            key={key}
            ref={(node) => { labelsRef.current[key] = node; }}
            className={`dimension-label${key === "local" ? " local hidden" : ""}`}
            data-dimension={key}
          />
        ))}
      </div>
      {loading && <div className="loading-overlay"><span />Loading DrivAer reference mesh</div>}
      {fallbackMessage && <div className="viewer-error" role="status">{fallbackMessage}</div>}
    </div>
  );
});

VehicleViewer.displayName = "VehicleViewer";
