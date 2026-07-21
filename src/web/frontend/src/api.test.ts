import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api, requestJson } from "./api";
import { NUMERIC_PARAMETER_NAMES, type DesignParameters } from "./types";

function design(): DesignParameters {
  const numeric = Object.fromEntries(
    NUMERIC_PARAMETER_NAMES.map((name, index) => [name, index]),
  ) as Pick<DesignParameters, (typeof NUMERIC_PARAMETER_NAMES)[number]>;
  return { ...numeric, CarRear: "Fastback", Wheels: "Open detailed" };
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Paragon API client", () => {
  it("sends typed designs as JSON", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ cd: 0.245 }));
    vi.stubGlobal("fetch", fetchMock);
    const parameters = design();

    await api.predict(parameters);

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/predict/parameters",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(parameters),
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
      }),
    );
  });

  it("turns FastAPI field validation into a readable ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      detail: [{ loc: ["body", "A_Car_Width"], msg: "Field required", type: "missing" }],
    }, 422)));

    const request = requestJson("/api/predict/parameters", { method: "POST", body: {} });

    await expect(request).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "A_Car_Width: Field required",
    });
  });

  it("preserves provider fallback information from nested HTTP errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({
      detail: {
        error: {
          code: "vertex_unavailable",
          message: "Vertex authentication unavailable",
          fallback_provider: "Local RandomForest",
        },
      },
    }, 503)));

    try {
      await api.testVertex({ parameters: design() });
      throw new Error("Expected request to fail");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect(error).toMatchObject({
        status: 503,
        code: "vertex_unavailable",
        fallbackProvider: "Local RandomForest",
      });
    }
  });

  it("uploads STL as multipart data without forcing a JSON content type", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ preview_points: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await api.uploadStl(new File(["solid car"], "car.stl", { type: "model/stl" }));

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.body).toBeInstanceOf(FormData);
    expect(new Headers(init.headers).has("Content-Type")).toBe(false);
  });
});
