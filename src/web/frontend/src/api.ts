import type {
  AnalysisResponse,
  ApiErrorPayload,
  CopilotRequest,
  CopilotResponse,
  DesignParameters,
  OptimizationRequest,
  OptimizationResponse,
  DemoCar,
  DemoCloud,
  DemoInference,
  ParameterSchema,
  PointNetDemoResponse,
  PredictionResponse,
  StatusResponse,
  StlPredictionResponse,
  VertexTestResponse,
} from "./types";

const DEFAULT_TIMEOUT_MS = 20_000;
const LONG_TIMEOUT_MS = 45_000;

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;
  readonly fallbackProvider?: string;
  readonly details?: unknown;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      fallbackProvider?: string;
      details?: unknown;
    } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status ?? 0;
    this.code = options.code;
    this.fallbackProvider = options.fallbackProvider;
    this.details = options.details;
  }
}

interface RequestOptions extends Omit<RequestInit, "body"> {
  body?: BodyInit | object;
  timeoutMs?: number;
}

function apiBase(): string {
  return String(import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
}

function validationMessage(detail: unknown): string | null {
  if (!Array.isArray(detail)) return null;
  const messages = detail
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const issue = item as { loc?: unknown; msg?: unknown };
      const path = Array.isArray(issue.loc)
        ? issue.loc.filter((part) => part !== "body").join(".")
        : "";
      const message = typeof issue.msg === "string" ? issue.msg : "Invalid value";
      return path ? `${path}: ${message}` : message;
    })
    .filter((item): item is string => Boolean(item));
  return messages.length ? messages.join("; ") : null;
}

function errorFromPayload(status: number, payload: unknown, statusText: string): ApiError {
  const body = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  const detail = body.detail;
  const nestedDetail =
    detail && typeof detail === "object" && !Array.isArray(detail)
      ? (detail as Record<string, unknown>)
      : undefined;
  const rawError = body.error ?? nestedDetail?.error;
  const error =
    rawError && typeof rawError === "object"
      ? (rawError as ApiErrorPayload)
      : undefined;
  const message =
    error?.message ??
    (typeof rawError === "string" ? rawError : undefined) ??
    validationMessage(detail) ??
    (typeof detail === "string" ? detail : undefined) ??
    statusText ??
    `Request failed (${status})`;

  return new ApiError(message, {
    status,
    code: error?.code,
    fallbackProvider: error?.fallback_provider,
    details: payload,
  });
}

export async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const controller = new AbortController();
  const { body, timeoutMs = DEFAULT_TIMEOUT_MS, headers, signal, ...init } = options;
  let timedOut = false;
  const abortFromCaller = () => controller.abort(signal?.reason);
  if (signal?.aborted) abortFromCaller();
  else signal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  const isJsonBody =
    body !== undefined &&
    !(body instanceof FormData) &&
    !(body instanceof Blob) &&
    !(body instanceof URLSearchParams) &&
    typeof body !== "string";

  try {
    const response = await fetch(`${apiBase()}${path}`, {
      ...init,
      headers: {
        ...(isJsonBody ? { "Content-Type": "application/json" } : {}),
        ...headers,
      },
      body: isJsonBody ? JSON.stringify(body) : (body as BodyInit | undefined),
      signal: controller.signal,
    });
    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    if (!response.ok) throw errorFromPayload(response.status, payload, response.statusText);
    return payload as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) {
      throw new ApiError("Request timed out. Please try again.", {
        code: "request_timeout",
      });
    }
    if (controller.signal.aborted) {
      throw new ApiError("Request was cancelled.", { code: "request_cancelled" });
    }
    throw new ApiError(error instanceof Error ? error.message : "Network request failed.", {
      code: "network_error",
    });
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromCaller);
  }
}

export const api = {
  getStatus: (signal?: AbortSignal) =>
    requestJson<StatusResponse>("/api/status", { signal }),

  getParameters: (signal?: AbortSignal) =>
    requestJson<ParameterSchema>("/api/parameters", { signal }),

  getDemoCars: (signal?: AbortSignal) =>
    requestJson<{ cars: DemoCar[] }>("/api/demo/pointnet/cars", { signal }),

  getDemoCloud: (id: string, signal?: AbortSignal) =>
    requestJson<DemoCloud>(`/api/demo/pointnet/cloud/${encodeURIComponent(id)}`, {
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    }),

  inferDemoCar: (id: string, points?: number, signal?: AbortSignal) =>
    requestJson<DemoInference>(
      `/api/demo/pointnet/infer/${encodeURIComponent(id)}` +
        (points ? `?points=${points}` : ""),
      {
        method: "POST",
        signal,
        timeoutMs: LONG_TIMEOUT_MS,
      },
    ),

  getPointNetDemo: (signal?: AbortSignal) =>
    requestJson<PointNetDemoResponse>("/api/demo/pointnet", {
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    }),

  predict: (parameters: DesignParameters, signal?: AbortSignal) =>
    requestJson<PredictionResponse>("/api/predict/parameters", {
      method: "POST",
      body: parameters,
      signal,
    }),

  analyze: (parameters: DesignParameters, signal?: AbortSignal) =>
    requestJson<AnalysisResponse>("/api/analyze/parameters", {
      method: "POST",
      body: parameters,
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    }),

  optimize: (request: OptimizationRequest, signal?: AbortSignal) =>
    requestJson<OptimizationResponse>("/api/optimize/parameters", {
      method: "POST",
      body: request,
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    }),

  copilot: (request: CopilotRequest, signal?: AbortSignal) =>
    requestJson<CopilotResponse>("/api/copilot", {
      method: "POST",
      body: request,
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    }),

  testVertex: (
    input?: DesignParameters | { parameters?: DesignParameters },
    signal?: AbortSignal,
  ) =>
    requestJson<VertexTestResponse>("/api/providers/vertex/test", {
      method: "POST",
      body: input && "CarRear" in input ? { parameters: input } : (input ?? {}),
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    }),

  uploadStl: (file: File, signal?: AbortSignal) => {
    const form = new FormData();
    form.append("file", file);
    return requestJson<StlPredictionResponse>("/api/predict", {
      method: "POST",
      body: form,
      signal,
      timeoutMs: LONG_TIMEOUT_MS,
    });
  },
};
