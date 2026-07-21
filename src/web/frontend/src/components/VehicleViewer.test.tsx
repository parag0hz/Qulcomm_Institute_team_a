import { act, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";

import type { DesignParameters, ParameterDefinition } from "../types";
import { NUMERIC_PARAMETER_NAMES } from "../types";
import { VehicleViewer } from "./VehicleViewer";

interface RendererDouble {
  loop: (() => void) | null;
  lastScene: THREE.Scene | null;
  disposed: boolean;
  contextLost: boolean;
}

const rendererState = vi.hoisted(() => ({
  instances: [] as RendererDouble[],
  initializationError: null as Error | null,
}));

vi.mock("three", async () => {
  const actual = await vi.importActual<typeof import("three")>("three");
  class WebGLRendererDouble implements RendererDouble {
    loop: (() => void) | null = null;
    lastScene: THREE.Scene | null = null;
    disposed = false;
    contextLost = false;
    outputColorSpace = actual.SRGBColorSpace;
    toneMapping = actual.NoToneMapping;
    toneMappingExposure = 1;
    shadowMap = { enabled: false, type: actual.PCFShadowMap };
    constructor() {
      if (rendererState.initializationError) throw rendererState.initializationError;
      rendererState.instances.push(this);
    }
    setPixelRatio() {}
    setSize() {}
    setAnimationLoop(loop: (() => void) | null) { this.loop = loop; }
    render(scene: THREE.Scene) { this.lastScene = scene; }
    dispose() { this.disposed = true; }
    forceContextLoss() { this.contextLost = true; }
  }
  return { ...actual, WebGLRenderer: WebGLRendererDouble };
});

vi.mock("three/examples/jsm/controls/OrbitControls.js", async () => {
  const module = await import("three");
  class OrbitControlsDouble {
    target = new module.Vector3();
    enableDamping = false;
    dampingFactor = 0;
    minDistance = 0;
    maxDistance = 0;
    maxPolarAngle = Math.PI;
    update() {}
    dispose() {}
  }
  return { OrbitControls: OrbitControlsDouble };
});

vi.mock("three/examples/jsm/loaders/GLTFLoader.js", async () => {
  const module = await import("three");
  class GLTFLoaderDouble {
    async loadAsync() {
      const scene = new module.Group();
      const body = new module.Mesh(
        new module.BoxGeometry(4.6, 1.8, 1.2, 3, 2, 2),
        new module.MeshStandardMaterial(),
      );
      body.name = "Body";
      body.position.z = 0.6;
      scene.add(body);
      const positions: Array<[string, number, number]> = [
        ["Wheel_FL", -1.4, -0.95],
        ["Wheel_FR", -1.4, 0.95],
        ["Wheel_RL", 1.4, -0.95],
        ["Wheel_RR", 1.4, 0.95],
      ];
      for (const [name, x, y] of positions) {
        const wheel = new module.Mesh(
          new module.CylinderGeometry(0.35, 0.35, 0.18, 12),
          new module.MeshStandardMaterial(),
        );
        wheel.name = name;
        wheel.position.set(x, y, 0.35);
        scene.add(wheel);
      }
      return { scene };
    }
  }
  return { GLTFLoader: GLTFLoaderDouble };
});

const numeric = Object.fromEntries(
  NUMERIC_PARAMETER_NAMES.map((name) => [name, 0]),
) as Pick<DesignParameters, (typeof NUMERIC_PARAMETER_NAMES)[number]>;
const design: DesignParameters = {
  ...numeric,
  CarRear: "Fastback",
  Wheels: "Open detailed",
};
const parameters: ParameterDefinition[] = NUMERIC_PARAMETER_NAMES.map((name) => ({
  name,
  label: name,
  group: "Test",
  min: -10,
  max: 10,
  default: 0,
  step: 1,
  high_impact: false,
}));

beforeEach(() => {
  rendererState.instances.length = 0;
  rendererState.initializationError = null;
});

describe("VehicleViewer", () => {
  it("loads named GLB parts, morphs from the source geometry, switches wheels, and disposes", async () => {
    const onStatusChange = vi.fn();
    const view = render(
      <VehicleViewer
        values={design}
        parameters={parameters}
        wheelTreatment="Open detailed"
        activeParameter="A_Car_Length"
        dimensionsVisible
        onStatusChange={onStatusChange}
      />,
    );

    await waitFor(() => expect(onStatusChange).toHaveBeenCalledWith(expect.objectContaining({
      referenceKind: "glb",
      datasetWheelCount: 4,
    })));
    const renderer = rendererState.instances[0];
    expect(renderer).toBeDefined();
    act(() => renderer.loop?.());
    const scene = renderer.lastScene!;
    const body = scene.getObjectByName("Body") as THREE.Mesh;
    const baseline = Array.from(body.geometry.getAttribute("position").array);

    view.rerender(
      <VehicleViewer
        values={{ ...design, A_Car_Length: 10 }}
        parameters={parameters}
        wheelTreatment="Open smooth"
        activeParameter="A_Car_Length"
        dimensionsVisible={false}
        onStatusChange={onStatusChange}
      />,
    );
    await waitFor(() => {
      const changed = Array.from(body.geometry.getAttribute("position").array);
      expect(changed).not.toEqual(baseline);
    });
    expect(scene.getObjectByName("Wheel_FL")?.visible).toBe(false);
    expect(scene.getObjectByName("Procedural_Open_smooth")?.visible).toBe(true);
    expect(scene.getObjectByName("Paragon_Dimensions")?.visible).toBe(false);

    view.unmount();
    expect(renderer.disposed).toBe(true);
    expect(renderer.contextLost).toBe(false);
  });

  it("survives the StrictMode setup-cleanup-setup cycle without forcing context loss", async () => {
    const onStatusChange = vi.fn();
    const view = render(
      <StrictMode>
        <VehicleViewer
          values={design}
          parameters={parameters}
          onStatusChange={onStatusChange}
        />
      </StrictMode>,
    );

    await waitFor(() => expect(onStatusChange).toHaveBeenCalledWith(expect.objectContaining({
      referenceKind: "glb",
    })));
    expect(rendererState.instances).toHaveLength(2);
    expect(rendererState.instances[0].disposed).toBe(true);
    expect(rendererState.instances.every((renderer) => !renderer.contextLost)).toBe(true);

    view.unmount();
    expect(rendererState.instances[1].disposed).toBe(true);
    expect(rendererState.instances[1].contextLost).toBe(false);
  });

  it("keeps the viewer mounted and reports a WebGL initialization failure", async () => {
    rendererState.initializationError = new TypeError("Cannot read properties of null (reading 'precision')");
    const onError = vi.fn();

    render(
      <VehicleViewer
        values={design}
        parameters={parameters}
        onError={onError}
      />,
    );

    expect(await screen.findByRole("status")).toHaveTextContent("WebGL could not be initialized");
    expect(screen.getByTestId("vehicle-viewer")).toBeInTheDocument();
    expect(onError).toHaveBeenCalledWith(expect.stringContaining("WebGL could not be initialized"));
  });
});
