import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api";
import type {
  AnalysisResponse,
  DesignParameters,
  ParameterSchema,
  PredictionResponse,
} from "../types";
import { NUMERIC_PARAMETER_NAMES } from "../types";
import { CopilotPanel } from "./CopilotPanel";
import { DesignControls } from "./DesignControls";
import { ResultsPanel } from "./ResultsPanel";

vi.mock("../api", () => ({
  api: {
    optimize: vi.fn(),
    copilot: vi.fn(),
    testVertex: vi.fn(),
  },
}));

const numericDesign = Object.fromEntries(
  NUMERIC_PARAMETER_NAMES.map((name, index) => [name, index + 1]),
) as Pick<DesignParameters, (typeof NUMERIC_PARAMETER_NAMES)[number]>;
const design: DesignParameters = {
  ...numericDesign,
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
    step: 0.5,
    high_impact: name === "E_Fenders_Arch_Offset",
  })),
  categories: {
    CarRear: ["Fastback", "Estateback", "Notchback"],
    Wheels: ["Open detailed", "Open smooth", "Closed smooth"],
  },
  valid_combinations: [{ CarRear: "Fastback", Wheels: "Open detailed" }],
  presets: [{ id: "median", name: "Dataset median", design }],
};

const prediction: PredictionResponse = {
  cd: 0.244,
  percentile: 31,
  level: "low",
  comparison: "Better than most observed designs.",
  provider: "Local RandomForest",
  domain_status: "outside",
  nearest_sample_distance: 0.3,
  uncertainty: { estimate: 0.01, lower: 0.224, upper: 0.264, basis: "test" },
  warnings: ["Prediction is outside the observed training domain."],
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
  drivers: [{
    name: "A_Car_Length",
    label: "Car length",
    minus_value: 18,
    plus_value: 20,
    minus_cd: 0.242,
    plus_cd: 0.246,
    minus_delta: -0.002,
    plus_delta: 0.002,
    impact: 0.002,
  }],
};

beforeEach(() => {
  vi.clearAllMocks();
});

describe("DesignControls", () => {
  it("updates numeric inputs and reports an unseen category combination", () => {
    const onParameterChange = vi.fn();
    render(<DesignControls
      schema={schema}
      design={{ ...design, CarRear: "Estateback", Wheels: "Open smooth" }}
      activeParameter="A_Car_Length"
      locked={[]}
      mode="parameters"
      stlBusy={false}
      onModeChange={vi.fn()}
      onParameterChange={onParameterChange}
      onCategoryChange={vi.fn()}
      onPreset={vi.fn()}
      onFocus={vi.fn()}
      onToggleLock={vi.fn()}
      onStlUpload={vi.fn()}
    />);

    fireEvent.change(screen.getByLabelText("Car length value"), { target: { value: "42" } });

    expect(onParameterChange).toHaveBeenCalledWith("A_Car_Length", 42);
    expect(screen.getByText(/combination was not present in training/i)).toBeInTheDocument();
  });
});

describe("ResultsPanel", () => {
  it("shows domain risk and requests bounded improvement directions", async () => {
    vi.mocked(api.optimize).mockResolvedValue({
      current_cd: prediction.cd,
      target_cd: 0.24,
      locked: ["CarRear", "Wheels"],
      message: "Surrogate guidance",
      recommendations: [{
        cd: 0.24,
        improvement: 0.004,
        distance: 0.08,
        domain_status: "inside",
        parameters: { ...design, A_Car_Length: 18 },
        changes: [{ name: "A_Car_Length", label: "Car length", from: 19, to: 18 }],
      }],
    });

    render(<ResultsPanel
      design={design}
      schema={schema}
      status={{ model_status: "RandomForest", model_metrics: { r2: 0.91, mae: 0.004 }, providers: { vertex: { available: false } } }}
      prediction={prediction}
      analysis={analysis}
      baselineCd={0.252}
      locks={["A_Car_Width"]}
      variants={[]}
      onFocusParameter={vi.fn()}
      onSetBaseline={vi.fn()}
      onPreview={vi.fn()}
      onApply={vi.fn()}
      onCancelPreview={vi.fn()}
      onLoadVariant={vi.fn()}
      onDeleteVariant={vi.fn()}
      onClearWorkspace={vi.fn()}
      onPrint={vi.fn()}
    />);

    expect(screen.getAllByText(/outside/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(prediction.warnings[0]).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Drivers" }));
    fireEvent.click(screen.getByRole("button", { name: "Find improvement directions" }));

    await waitFor(() => expect(api.optimize).toHaveBeenCalledWith({
      parameters: design,
      target_cd: 0.24,
      locked: ["A_Car_Width"],
    }, expect.any(AbortSignal)));
    expect(await screen.findByText(/Direction 1 · Cd 0.2400/)).toBeInTheDocument();
  });
});

describe("CopilotPanel", () => {
  it("sends the current design and renders grounded provider metadata", async () => {
    vi.mocked(api.copilot).mockResolvedValue({
      answer: "Review the diffuser and fender arch first.",
      provider: "grounded_local",
      model: null,
      evidence: {
        prediction,
        top_drivers: analysis.drivers,
        goal_search: null,
      },
      disclaimer: "Validate with CFD.",
    });
    render(<CopilotPanel design={design} />);

    fireEvent.click(screen.getByRole("button", { name: "Review top drivers" }));

    expect(await screen.findByText("Review the diffuser and fender arch first.")).toBeInTheDocument();
    expect(screen.getByText("grounded_local")).toBeInTheDocument();
    expect(api.copilot).toHaveBeenCalledWith(expect.objectContaining({
      message: "Which three parameters should I review first and why?",
      parameters: design,
    }));
  });

  it("bounds prior LLM answers to the FastAPI history contract", async () => {
    const longAnswer = "A".repeat(2200);
    vi.mocked(api.copilot)
      .mockResolvedValueOnce({
        answer: longAnswer,
        provider: "openai",
        model: "test-model",
        evidence: { prediction, top_drivers: [], goal_search: null },
        disclaimer: "Validate with CFD.",
      })
      .mockResolvedValueOnce({
        answer: "Second answer",
        provider: "grounded_local",
        model: null,
        evidence: { prediction, top_drivers: [], goal_search: null },
        disclaimer: "Validate with CFD.",
      });
    render(<CopilotPanel design={design} />);

    fireEvent.click(screen.getByRole("button", { name: "Explain current Cd" }));
    expect(await screen.findByText(longAnswer)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Review top drivers" }));
    expect(await screen.findByText("Second answer")).toBeInTheDocument();

    const secondRequest = vi.mocked(api.copilot).mock.calls[1]?.[0];
    expect(secondRequest?.history?.every((item) => item.content.length <= 2000)).toBe(true);
  });
});
