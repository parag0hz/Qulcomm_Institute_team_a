import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { useWorkspaceStore } from "./store";
import type {
  AnalysisResponse,
  DesignParameters,
  ParameterSchema,
  PredictionResponse,
  StatusResponse,
} from "./types";
import { NUMERIC_PARAMETER_NAMES } from "./types";

const apiMocks = vi.hoisted(() => ({
  getParameters: vi.fn(),
  getStatus: vi.fn(),
  predict: vi.fn(),
  analyze: vi.fn(),
  uploadStl: vi.fn(),
  optimize: vi.fn(),
  copilot: vi.fn(),
  testVertex: vi.fn(),
}));
const viewerMockState = vi.hoisted(() => ({ shouldThrow: false }));

vi.mock("./api", () => ({ api: apiMocks }));
vi.mock("./components/VehicleViewer", async () => {
  const React = await import("react");
  return {
    VehicleViewer: React.forwardRef(function MockVehicleViewer(
      props: { values: DesignParameters; wheelTreatment: string; importedPoints?: readonly unknown[] },
      ref: React.ForwardedRef<unknown>,
    ) {
      if (viewerMockState.shouldThrow) throw new Error("Unexpected viewer failure");
      React.useImperativeHandle(ref, () => ({
        setView: vi.fn(),
        resetView: vi.fn(),
        captureThumbnail: () => "data:image/jpeg;base64,test",
        showReference: vi.fn(),
      }));
      return <div data-testid="vehicle-viewer" data-length={props.values.A_Car_Length} data-wheel={props.wheelTreatment} data-imported={Boolean(props.importedPoints?.length)} />;
    }),
  };
});

const numeric = Object.fromEntries(
  NUMERIC_PARAMETER_NAMES.map((name, index) => [name, index + 1]),
) as Pick<DesignParameters, (typeof NUMERIC_PARAMETER_NAMES)[number]>;
const initialDesign: DesignParameters = {
  ...numeric,
  CarRear: "Fastback",
  Wheels: "Open detailed",
};

const schema: ParameterSchema = {
  parameters: NUMERIC_PARAMETER_NAMES.map((name, index) => ({
    name,
    label: name === "A_Car_Length" ? "Car length" : name.replaceAll("_", " "),
    group: "Body proportions",
    min: 0,
    max: 100,
    default: index + 1,
    step: 1,
    high_impact: name === "E_Fenders_Arch_Offset",
  })),
  categories: {
    CarRear: ["Fastback", "Estateback", "Notchback"],
    Wheels: ["Open detailed", "Open smooth", "Closed smooth"],
  },
  valid_combinations: [{ CarRear: "Fastback", Wheels: "Open detailed" }],
  presets: [{ id: "median", name: "Dataset median", design: initialDesign }],
};

const prediction: PredictionResponse = {
  cd: 0.244,
  percentile: 31,
  level: "low",
  comparison: "Below dataset median.",
  provider: "Local RandomForest",
  domain_status: "inside",
  nearest_sample_distance: 0.08,
  uncertainty: { estimate: 0.01, lower: 0.224, upper: 0.264, basis: "test" },
  warnings: [],
  model: { name: "RandomForest", status: "trained", confidence: "medium" },
  dataset: {
    sample_count: 4165,
    cd_min: 0.2,
    cd_max: 0.32,
    cd_mean: 0.254,
    cd_median: 0.252,
    cd_p25: 0.237,
    cd_p75: 0.27,
    feature_count: 23,
    feature_columns: [...NUMERIC_PARAMETER_NAMES],
  },
};

const analysis: AnalysisResponse = {
  base_cd: prediction.cd,
  provider: prediction.provider,
  warnings: [],
  drivers: [],
};

const status: StatusResponse = {
  product: "paragon",
  name: "Paragon Vehicle Design Studio",
  model_status: "RandomForest",
  trained_model_connected: true,
  model_metrics: { r2: 0.91, mae: 0.004 },
  providers: { active: "Local RandomForest", local: { available: true }, vertex: { available: false } },
  copilot: { configured: false, provider: "grounded_local", model: null, message: "Local explainer" },
  input_schema: { numeric_features: [...NUMERIC_PARAMETER_NAMES], categorical_features: ["CarRear", "Wheels"] },
  dataset: prediction.dataset,
};

beforeEach(() => {
  viewerMockState.shouldThrow = false;
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
  vi.clearAllMocks();
  apiMocks.getParameters.mockResolvedValue(schema);
  apiMocks.getStatus.mockResolvedValue(status);
  apiMocks.predict.mockResolvedValue(prediction);
  apiMocks.analyze.mockResolvedValue(analysis);
});

describe("Paragon App integration", () => {
  it("keeps the dashboard visible when the 3D viewer throws unexpectedly", async () => {
    viewerMockState.shouldThrow = true;
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<App />);

    expect(await screen.findByText("Explore your concept")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent("3D preview unavailable");
    expect(screen.getByText("Aerodynamic result")).toBeInTheDocument();
  });

  it("loads the schema and sends the same slider design to the viewer and prediction API", async () => {
    render(<App />);

    expect(await screen.findByText("Explore your concept")).toBeInTheDocument();
    expect(await screen.findByText("0.2440")).toBeInTheDocument();
    apiMocks.predict.mockClear();

    const lengthInput = screen.getByLabelText("Car length value");
    fireEvent.change(lengthInput, { target: { value: "42" } });
    fireEvent.blur(lengthInput);

    expect(screen.getByTestId("vehicle-viewer")).toHaveAttribute("data-length", "42");
    expect(screen.getByRole("button", { name: "Save variant" })).toBeDisabled();
    await waitFor(() => expect(apiMocks.predict).toHaveBeenCalledWith(
      expect.objectContaining({ A_Car_Length: 42 }),
      expect.any(AbortSignal),
    ), { timeout: 1500 });
  });

  it("uses the uploaded STL result throughout the workspace instead of the parametric Cd", async () => {
    apiMocks.uploadStl.mockResolvedValue({
      cd: 0.271,
      percentile: 76,
      level: "high",
      comparison: "Above the dataset median.",
      model: { name: "geometry_fallback", status: "not_trained", confidence: "low" },
      dataset: prediction.dataset,
      preview_points: [[0, 0, 0], [1, 0, 0]],
      file: { name: "concept.stl", size_bytes: 128 },
      mesh: { triangle_count: 2 },
    });
    render(<App />);
    expect(await screen.findByText("0.2440")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Import STL" }));
    const fileInput = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(fileInput).not.toBeNull();
    fireEvent.change(fileInput!, {
      target: { files: [new File(["solid concept"], "concept.stl", { type: "model/stl" })] },
    });

    expect((await screen.findAllByText("0.2710")).length).toBeGreaterThan(0);
    expect(screen.getByText("Imported STL · untrained fallback")).toBeInTheDocument();
    expect(screen.getByTestId("vehicle-viewer")).toHaveAttribute("data-imported", "true");
    expect(screen.queryByRole("button", { name: "Set current design as baseline" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save variant" })).toBeDisabled();
  });
});
