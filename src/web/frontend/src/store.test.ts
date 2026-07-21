import { beforeEach, describe, expect, it } from "vitest";

import {
  MAX_VARIANTS,
  WORKSPACE_V1_KEY,
  WORKSPACE_V2_KEY,
  selectCanRedo,
  selectCanUndo,
  useWorkspaceStore,
} from "./store";
import {
  NUMERIC_PARAMETER_NAMES,
  type DesignParameters,
  type Recommendation,
} from "./types";

function design(offset = 0): DesignParameters {
  const numeric = Object.fromEntries(
    NUMERIC_PARAMETER_NAMES.map((name, index) => [name, index + offset]),
  ) as Pick<DesignParameters, (typeof NUMERIC_PARAMETER_NAMES)[number]>;
  return {
    ...numeric,
    CarRear: "Fastback",
    Wheels: "Open detailed",
  };
}

function resetStore(): void {
  localStorage.clear();
  useWorkspaceStore.setState({
    current: null,
    baseline: null,
    history: [],
    historyIndex: -1,
    locks: [],
    variants: [],
    designName: "Untitled concept",
    recommendationPreview: null,
  });
}

function recommendation(parameters: DesignParameters): Recommendation {
  return {
    cd: 0.24,
    improvement: 0.01,
    distance: 0.08,
    domain_status: "inside",
    parameters,
    changes: [],
  };
}

beforeEach(resetStore);

describe("workspace history", () => {
  it("initializes a baseline and supports undo/redo", () => {
    const initial = design();
    useWorkspaceStore.getState().initializeDesign(initial, 0.252);
    useWorkspaceStore.getState().updateNumericParameter("A_Car_Length", 100);

    expect(selectCanUndo(useWorkspaceStore.getState())).toBe(true);
    expect(useWorkspaceStore.getState().current?.A_Car_Length).toBe(100);

    useWorkspaceStore.getState().undo();
    expect(useWorkspaceStore.getState().current?.A_Car_Length).toBe(initial.A_Car_Length);
    expect(selectCanRedo(useWorkspaceStore.getState())).toBe(true);

    useWorkspaceStore.getState().redo();
    expect(useWorkspaceStore.getState().current?.A_Car_Length).toBe(100);
    expect(useWorkspaceStore.getState().baseline?.cd).toBe(0.252);
  });

  it("drops the redo branch after a new edit", () => {
    useWorkspaceStore.getState().initializeDesign(design());
    useWorkspaceStore.getState().updateNumericParameter("A_Car_Length", 100);
    useWorkspaceStore.getState().updateNumericParameter("A_Car_Length", 200);
    useWorkspaceStore.getState().undo();
    useWorkspaceStore.getState().updateNumericParameter("A_Car_Length", 300);

    expect(selectCanRedo(useWorkspaceStore.getState())).toBe(false);
    expect(useWorkspaceStore.getState().history).toHaveLength(3);
    expect(useWorkspaceStore.getState().current?.A_Car_Length).toBe(300);
  });
});

describe("variants, locks, and recommendation preview", () => {
  it("persists locks and caps variants at five", () => {
    useWorkspaceStore.getState().initializeDesign(design());
    useWorkspaceStore.getState().toggleLock("A_Car_Width");
    for (let index = 0; index < MAX_VARIANTS + 1; index += 1) {
      useWorkspaceStore.getState().saveVariant({ name: `Candidate ${index}` });
    }

    expect(useWorkspaceStore.getState().locks).toEqual(["A_Car_Width"]);
    expect(useWorkspaceStore.getState().variants).toHaveLength(MAX_VARIANTS);
    expect(useWorkspaceStore.getState().saveVariant()).toBeNull();
  });

  it("previews without history and restores or commits explicitly", () => {
    const initial = design();
    const candidate = design(10);
    useWorkspaceStore.getState().initializeDesign(initial);
    useWorkspaceStore.getState().previewRecommendation(recommendation(candidate));

    expect(useWorkspaceStore.getState().current).toEqual(candidate);
    expect(useWorkspaceStore.getState().history).toHaveLength(1);

    useWorkspaceStore.getState().cancelRecommendation();
    expect(useWorkspaceStore.getState().current).toEqual(initial);

    useWorkspaceStore.getState().previewRecommendation(recommendation(candidate));
    useWorkspaceStore.getState().applyRecommendation();
    expect(useWorkspaceStore.getState().current).toEqual(candidate);
    expect(useWorkspaceStore.getState().history).toHaveLength(2);
    expect(useWorkspaceStore.getState().recommendationPreview).toBeNull();
  });
});

describe("v1 workspace migration", () => {
  it("moves the legacy payload to v2 and creates recoverable history", async () => {
    const current = design(2);
    localStorage.setItem(WORKSPACE_V1_KEY, JSON.stringify({
      current,
      variants: [{
        id: "legacy-a",
        name: "Variant A",
        design: current,
        cd: 0.243,
        baselineDelta: -0.009,
        thumbnail: "data:image/jpeg;base64,abc",
        savedAt: "2026-07-15T00:00:00.000Z",
      }],
      locked: ["A_Car_Width", "unknown"],
      baselineCd: 0.252,
      designName: "Legacy concept",
    }));
    localStorage.removeItem(WORKSPACE_V2_KEY);

    await useWorkspaceStore.persist.rehydrate();
    const state = useWorkspaceStore.getState();

    expect(state.current).toEqual(current);
    expect(state.baseline).toEqual({ design: current, cd: 0.252 });
    expect(state.history).toEqual([current]);
    expect(state.historyIndex).toBe(0);
    expect(state.locks).toEqual(["A_Car_Width"]);
    expect(state.variants[0]?.id).toBe("legacy-a");
    expect(state.designName).toBe("Legacy concept");
    expect(localStorage.getItem(WORKSPACE_V1_KEY)).toBeNull();
    expect(localStorage.getItem(WORKSPACE_V2_KEY)).not.toBeNull();
  });
});
