import { create } from "zustand";
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware";

import {
  NUMERIC_PARAMETER_NAMES,
  type Baseline,
  type CarRear,
  type DesignParameters,
  type NumericParameterName,
  type Recommendation,
  type RecommendationPreview,
  type Variant,
  type WheelTreatment,
} from "./types";

export const WORKSPACE_V1_KEY = "paragon.workspace.v1";
export const WORKSPACE_V2_KEY = "paragon.workspace.v2";
export const MAX_HISTORY = 50;
export const MAX_VARIANTS = 5;

interface SaveVariantInput {
  name?: string;
  cd?: number | null;
  baselineDelta?: number | null;
  thumbnail?: string;
}

interface DesignUpdateOptions {
  record?: boolean;
}

interface PersistedWorkspace {
  current: DesignParameters | null;
  baseline: Baseline | null;
  history: DesignParameters[];
  historyIndex: number;
  locks: NumericParameterName[];
  variants: Variant[];
  designName: string;
}

export interface WorkspaceState extends PersistedWorkspace {
  recommendationPreview: RecommendationPreview | null;
  initializeDesign: (design: DesignParameters, baselineCd?: number | null) => void;
  setCurrentDesign: (design: DesignParameters, options?: DesignUpdateOptions) => void;
  updateNumericParameter: (
    name: NumericParameterName,
    value: number,
    options?: DesignUpdateOptions,
  ) => void;
  updateConfiguration: (
    configuration: Partial<Pick<DesignParameters, "CarRear" | "Wheels">>,
    options?: DesignUpdateOptions,
  ) => void;
  setBaseline: (design: DesignParameters, cd?: number | null) => void;
  setBaselineCd: (cd: number | null) => void;
  setDesignName: (name: string) => void;
  undo: () => void;
  redo: () => void;
  toggleLock: (name: NumericParameterName) => void;
  saveVariant: (input?: SaveVariantInput) => Variant | null;
  deleteVariant: (id: string) => void;
  loadVariant: (id: string) => void;
  previewRecommendation: (recommendation: Recommendation) => void;
  applyRecommendation: (recommendation?: Recommendation) => void;
  cancelRecommendation: () => void;
  clearWorkspace: (initialDesign?: DesignParameters) => void;
}

const numericNames = new Set<string>(NUMERIC_PARAMETER_NAMES);
const carRears = new Set<CarRear>(["Fastback", "Estateback", "Notchback"]);
const wheels = new Set<WheelTreatment>([
  "Open detailed",
  "Open smooth",
  "Closed smooth",
]);

const cloneDesign = (design: DesignParameters): DesignParameters => ({ ...design });

function normalizeDesign(value: unknown): DesignParameters | null {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  if (!carRears.has(source.CarRear as CarRear) || !wheels.has(source.Wheels as WheelTreatment)) {
    return null;
  }
  const numeric = {} as Record<NumericParameterName, number>;
  for (const name of NUMERIC_PARAMETER_NAMES) {
    const parsed = Number(source[name]);
    if (!Number.isFinite(parsed)) return null;
    numeric[name] = parsed;
  }
  return {
    ...numeric,
    CarRear: source.CarRear as CarRear,
    Wheels: source.Wheels as WheelTreatment,
  };
}

function designsEqual(left: DesignParameters, right: DesignParameters): boolean {
  return (
    left.CarRear === right.CarRear &&
    left.Wheels === right.Wheels &&
    NUMERIC_PARAMETER_NAMES.every((name) => left[name] === right[name])
  );
}

function finiteOrNull(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeLocks(value: unknown): NumericParameterName[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((name): name is NumericParameterName =>
    typeof name === "string" && numericNames.has(name),
  ))];
}

function normalizeVariant(value: unknown): Variant | null {
  if (!value || typeof value !== "object") return null;
  const source = value as Record<string, unknown>;
  const design = normalizeDesign(source.design);
  if (!design) return null;
  return {
    id: typeof source.id === "string" && source.id ? source.id : createId(),
    name: typeof source.name === "string" && source.name ? source.name.slice(0, 80) : "Variant",
    design,
    cd: finiteOrNull(source.cd),
    baselineDelta: finiteOrNull(source.baselineDelta),
    thumbnail: typeof source.thumbnail === "string" ? source.thumbnail : "",
    savedAt: typeof source.savedAt === "string" ? source.savedAt : new Date().toISOString(),
  };
}

function normalizeVariants(value: unknown): Variant[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(normalizeVariant)
    .filter((variant): variant is Variant => Boolean(variant))
    .slice(0, MAX_VARIANTS);
}

function normalizeHistory(value: unknown): DesignParameters[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(normalizeDesign)
    .filter((design): design is DesignParameters => Boolean(design))
    .slice(-MAX_HISTORY);
}

function createId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function appendHistory(
  history: DesignParameters[],
  historyIndex: number,
  design: DesignParameters,
): { history: DesignParameters[]; historyIndex: number } {
  const active = history[historyIndex];
  if (active && designsEqual(active, design)) return { history, historyIndex };
  const next = [...history.slice(0, historyIndex + 1), cloneDesign(design)].slice(-MAX_HISTORY);
  return { history: next, historyIndex: next.length - 1 };
}

const memoryValues = new Map<string, string>();
const memoryStorage: Storage = {
  get length() { return memoryValues.size; },
  clear: () => memoryValues.clear(),
  getItem: (key) => memoryValues.get(key) ?? null,
  key: (index) => [...memoryValues.keys()][index] ?? null,
  removeItem: (key) => { memoryValues.delete(key); },
  setItem: (key, value) => { memoryValues.set(key, value); },
};

function browserStorage(): Storage {
  return typeof window === "undefined" ? memoryStorage : window.localStorage;
}

export function migrateLegacyWorkspace(storage: Storage = browserStorage()): boolean {
  if (storage.getItem(WORKSPACE_V2_KEY)) return false;
  const raw = storage.getItem(WORKSPACE_V1_KEY);
  if (!raw) return false;
  try {
    const legacy = JSON.parse(raw) as Record<string, unknown>;
    const current = normalizeDesign(legacy.current);
    const baselineCd = finiteOrNull(legacy.baselineCd);
    const state: PersistedWorkspace = {
      current,
      baseline: current && baselineCd !== null
        ? { design: cloneDesign(current), cd: baselineCd }
        : null,
      history: current ? [cloneDesign(current)] : [],
      historyIndex: current ? 0 : -1,
      locks: normalizeLocks(legacy.locked),
      variants: normalizeVariants(legacy.variants),
      designName:
        typeof legacy.designName === "string" && legacy.designName
          ? legacy.designName.slice(0, 120)
          : "Untitled concept",
    };
    storage.setItem(WORKSPACE_V2_KEY, JSON.stringify({ state, version: 2 }));
    storage.removeItem(WORKSPACE_V1_KEY);
    return true;
  } catch {
    storage.removeItem(WORKSPACE_V1_KEY);
    return false;
  }
}

const migrationAwareStorage: StateStorage = {
  getItem: (name) => {
    const storage = browserStorage();
    if (name === WORKSPACE_V2_KEY) migrateLegacyWorkspace(storage);
    return storage.getItem(name);
  },
  setItem: (name, value) => browserStorage().setItem(name, value),
  removeItem: (name) => browserStorage().removeItem(name),
};

function mergePersistedState(
  persisted: unknown,
  currentState: WorkspaceState,
): WorkspaceState {
  if (!persisted || typeof persisted !== "object") return currentState;
  const source = persisted as Partial<PersistedWorkspace>;
  const current = normalizeDesign(source.current) ?? currentState.current;
  const history = normalizeHistory(source.history);
  const requestedIndex = Number(source.historyIndex);
  const historyIndex = history.length
    ? Math.min(Math.max(Number.isInteger(requestedIndex) ? requestedIndex : history.length - 1, 0), history.length - 1)
    : -1;
  const baselineDesign = normalizeDesign(source.baseline?.design);
  const baseline = baselineDesign
    ? { design: baselineDesign, cd: finiteOrNull(source.baseline?.cd) }
    : null;
  return {
    ...currentState,
    current,
    baseline,
    history: history.length ? history : current ? [cloneDesign(current)] : [],
    historyIndex: history.length ? historyIndex : current ? 0 : -1,
    locks: normalizeLocks(source.locks),
    variants: normalizeVariants(source.variants),
    designName:
      typeof source.designName === "string" && source.designName
        ? source.designName.slice(0, 120)
        : currentState.designName,
    recommendationPreview: null,
  };
}

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
      current: null,
      baseline: null,
      history: [],
      historyIndex: -1,
      locks: [],
      variants: [],
      designName: "Untitled concept",
      recommendationPreview: null,

      initializeDesign: (design, baselineCd = null) => set((state) => {
        if (state.current) {
          return state.baseline
            ? state
            : { baseline: { design: cloneDesign(design), cd: baselineCd } };
        }
        const initial = cloneDesign(design);
        return {
          current: initial,
          baseline: { design: cloneDesign(design), cd: baselineCd },
          history: [cloneDesign(initial)],
          historyIndex: 0,
        };
      }),

      setCurrentDesign: (design, options = {}) => set((state) => {
        const current = cloneDesign(design);
        const record = options.record ?? true;
        return {
          current,
          recommendationPreview: null,
          ...(record ? appendHistory(state.history, state.historyIndex, current) : {}),
        };
      }),

      updateNumericParameter: (name, value, options = {}) => set((state) => {
        if (!state.current || !Number.isFinite(value)) return state;
        const current = { ...state.current, [name]: value };
        const record = options.record ?? true;
        return {
          current,
          recommendationPreview: null,
          ...(record ? appendHistory(state.history, state.historyIndex, current) : {}),
        };
      }),

      updateConfiguration: (configuration, options = {}) => set((state) => {
        if (!state.current) return state;
        const current = { ...state.current, ...configuration };
        const record = options.record ?? true;
        return {
          current,
          recommendationPreview: null,
          ...(record ? appendHistory(state.history, state.historyIndex, current) : {}),
        };
      }),

      setBaseline: (design, cd = null) => set({
        baseline: { design: cloneDesign(design), cd },
      }),

      setBaselineCd: (cd) => set((state) => ({
        baseline: state.baseline
          ? { ...state.baseline, cd }
          : state.current
            ? { design: cloneDesign(state.current), cd }
            : null,
      })),

      setDesignName: (designName) => set({
        designName: designName.trim().slice(0, 120) || "Untitled concept",
      }),

      undo: () => set((state) => {
        if (state.historyIndex <= 0) return state;
        const historyIndex = state.historyIndex - 1;
        return {
          historyIndex,
          current: cloneDesign(state.history[historyIndex]),
          recommendationPreview: null,
        };
      }),

      redo: () => set((state) => {
        if (state.historyIndex >= state.history.length - 1) return state;
        const historyIndex = state.historyIndex + 1;
        return {
          historyIndex,
          current: cloneDesign(state.history[historyIndex]),
          recommendationPreview: null,
        };
      }),

      toggleLock: (name) => set((state) => ({
        locks: state.locks.includes(name)
          ? state.locks.filter((item) => item !== name)
          : [...state.locks, name],
      })),

      saveVariant: (input = {}) => {
        const state = get();
        if (!state.current || state.variants.length >= MAX_VARIANTS) return null;
        const variant: Variant = {
          id: createId(),
          name: input.name?.trim().slice(0, 80) || `Variant ${String.fromCharCode(65 + state.variants.length)}`,
          design: cloneDesign(state.current),
          cd: finiteOrNull(input.cd),
          baselineDelta: finiteOrNull(input.baselineDelta),
          thumbnail: input.thumbnail ?? "",
          savedAt: new Date().toISOString(),
        };
        set({ variants: [...state.variants, variant] });
        return variant;
      },

      deleteVariant: (id) => set((state) => ({
        variants: state.variants.filter((variant) => variant.id !== id),
      })),

      loadVariant: (id) => set((state) => {
        const variant = state.variants.find((item) => item.id === id);
        if (!variant) return state;
        const current = cloneDesign(variant.design);
        return {
          current,
          recommendationPreview: null,
          ...appendHistory(state.history, state.historyIndex, current),
        };
      }),

      previewRecommendation: (recommendation) => set((state) => {
        if (!state.current) return state;
        return {
          recommendationPreview: {
            original: state.recommendationPreview?.original ?? cloneDesign(state.current),
            recommendation,
          },
          current: cloneDesign(recommendation.parameters),
        };
      }),

      applyRecommendation: (recommendation) => set((state) => {
        const current = recommendation
          ? cloneDesign(recommendation.parameters)
          : state.current && cloneDesign(state.current);
        if (!current) return state;
        return {
          current,
          recommendationPreview: null,
          ...appendHistory(state.history, state.historyIndex, current),
        };
      }),

      cancelRecommendation: () => set((state) => {
        if (!state.recommendationPreview) return state;
        return {
          current: cloneDesign(state.recommendationPreview.original),
          recommendationPreview: null,
        };
      }),

      clearWorkspace: (initialDesign) => {
        browserStorage().removeItem(WORKSPACE_V1_KEY);
        const current = initialDesign ? cloneDesign(initialDesign) : null;
        set({
          current,
          baseline: current ? { design: cloneDesign(current), cd: null } : null,
          history: current ? [cloneDesign(current)] : [],
          historyIndex: current ? 0 : -1,
          locks: [],
          variants: [],
          designName: "Untitled concept",
          recommendationPreview: null,
        });
      },
    }),
    {
      name: WORKSPACE_V2_KEY,
      version: 2,
      storage: createJSONStorage(() => migrationAwareStorage),
      merge: mergePersistedState,
      partialize: (state): PersistedWorkspace => ({
        current: state.current,
        baseline: state.baseline,
        history: state.history,
        historyIndex: state.historyIndex,
        locks: state.locks,
        variants: state.variants,
        designName: state.designName,
      }),
    },
  ),
);

export const selectCanUndo = (state: WorkspaceState): boolean => state.historyIndex > 0;
export const selectCanRedo = (state: WorkspaceState): boolean =>
  state.historyIndex >= 0 && state.historyIndex < state.history.length - 1;
