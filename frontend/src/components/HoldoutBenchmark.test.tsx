import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HoldoutBenchmark } from "./HoldoutBenchmark";
import { api } from "../api";
import type { PointNetDemoResponse } from "../types";

const AVAILABLE: PointNetDemoResponse = {
  available: true,
  point_count: 2048,
  mean_error_counts: 3.64,
  items: [
    {
      id: "N_S_WW_WM_430",
      body_type: "Notchback",
      true_cd: 0.24498,
      cd: 0.2442,
      raw_cd: 0.2442,
      trusted: true,
      warnings: [],
      error_counts: 0.76,
    },
    {
      id: "F_S_WWS_WM_347",
      body_type: "Fastback",
      true_cd: 0.26195,
      cd: null,
      raw_cd: 0.9123,
      trusted: false,
      warnings: ["Predicted Cd falls outside the range observed during training."],
      error_counts: 65.04,
    },
  ],
};

describe("HoldoutBenchmark", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("stays out of the DOM until it is opened", () => {
    const spy = vi.spyOn(api, "getPointNetDemo");
    const { container } = render(<HoldoutBenchmark open={false} onClose={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
    expect(spy).not.toHaveBeenCalled();
  });

  it("runs a live prediction on open and shows every held-out car", async () => {
    vi.spyOn(api, "getPointNetDemo").mockResolvedValue(AVAILABLE);
    render(<HoldoutBenchmark open onClose={vi.fn()} />);

    expect(await screen.findByText("3.64")).toBeInTheDocument();
    expect(screen.getByText("N_S_WW_WM_430")).toBeInTheDocument();
    expect(screen.getByText("0.2442")).toBeInTheDocument();
    // 최악 사례도 숨기지 않는다 — 체리피킹이 아님을 보이는 부분.
    expect(screen.getByText("F_S_WWS_WM_347")).toBeInTheDocument();
    expect(screen.getByText(/all 2 held-out cars shown/i)).toBeInTheDocument();
  });

  it("hides the number when the model reports an out-of-distribution shape", async () => {
    vi.spyOn(api, "getPointNetDemo").mockResolvedValue(AVAILABLE);
    render(<HoldoutBenchmark open onClose={vi.fn()} />);

    await screen.findByText("N_S_WW_WM_430");
    expect(screen.getByText("out of distribution")).toBeInTheDocument();
    expect(screen.queryByText("0.9123")).not.toBeInTheDocument();
  });

  it("explains itself when the demo assets are missing instead of failing", async () => {
    vi.spyOn(api, "getPointNetDemo").mockResolvedValue({
      available: false,
      reason: "PointNet demo assets are not installed.",
      items: [],
    });
    render(<HoldoutBenchmark open onClose={vi.fn()} />);

    expect(await screen.findByText(/assets are not installed/i)).toBeInTheDocument();
  });

  it("surfaces a request failure without crashing", async () => {
    vi.spyOn(api, "getPointNetDemo").mockRejectedValue(new Error("Network unreachable"));
    render(<HoldoutBenchmark open onClose={vi.fn()} />);

    expect(await screen.findByText("Network unreachable")).toBeInTheDocument();
  });

  it("re-runs the prediction on demand", async () => {
    const spy = vi.spyOn(api, "getPointNetDemo").mockResolvedValue(AVAILABLE);
    render(<HoldoutBenchmark open onClose={vi.fn()} />);
    await screen.findByText("3.64");

    fireEvent.click(screen.getByRole("button", { name: /run prediction again/i }));
    await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
  });

  it("closes on Escape", async () => {
    vi.spyOn(api, "getPointNetDemo").mockResolvedValue(AVAILABLE);
    const onClose = vi.fn();
    render(<HoldoutBenchmark open onClose={onClose} />);
    await screen.findByText("3.64");

    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });
});
