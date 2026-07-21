export const NUMERIC_PARAMETER_NAMES = [
  "B_Ramp_Angle",
  "B_Diffusor_Angle",
  "B_Trunklid_Angle",
  "C_Side_Mirrors_Rotation",
  "D_Rear_Window_Inclination",
  "D_Winscreen_Inclination",
  "C_Side_Mirrors_Translate_X",
  "C_Side_Mirrors_Translate_Z",
  "D_Winscreen_Length",
  "D_Rear_Window_Length",
  "E_A_B_C_Pillar_Thickness",
  "G_Trunklid_Curvature",
  "G_Trunklid_Length",
  "H_Front_Bumper_Curvature",
  "H_Front_Bumper_Length",
  "F_Door_Handles_Thickness",
  "F_Door_Handles_Z_Position",
  "E_Fenders_Arch_Offset",
  "A_Car_Length",
  "F_Door_Handles_X_Position",
  "A_Car_Width",
  "A_Car_Roof_Height",
  "A_Car_Green_House_Angle",
] as const;

export type NumericParameterName = (typeof NUMERIC_PARAMETER_NAMES)[number];
export type CarRear = "Fastback" | "Estateback" | "Notchback";
export type WheelTreatment = "Open detailed" | "Open smooth" | "Closed smooth";
export type DomainStatus = "inside" | "edge" | "outside";

export type NumericDesignValues = Record<NumericParameterName, number>;
export type DesignParameters = NumericDesignValues & {
  CarRear: CarRear;
  Wheels: WheelTreatment;
};

export interface ParameterDefinition {
  name: NumericParameterName;
  label: string;
  group: string;
  min: number;
  max: number;
  default: number;
  step: number;
  high_impact: boolean;
}

export interface DesignPreset {
  id: string;
  name: string;
  design: DesignParameters;
}

export interface ParameterSchema {
  parameters: ParameterDefinition[];
  categories: {
    CarRear: CarRear[];
    Wheels: WheelTreatment[];
  };
  valid_combinations: Array<Pick<DesignParameters, "CarRear" | "Wheels">>;
  presets: DesignPreset[];
}

export interface DatasetSummary {
  sample_count: number;
  cd_min: number;
  cd_max: number;
  cd_mean: number;
  cd_median: number;
  cd_p25: number;
  cd_p75: number;
  feature_count: number;
  feature_columns: NumericParameterName[];
}

export interface ModelMetrics {
  r2?: number;
  mae?: number;
  [key: string]: number | string | undefined;
}

export interface ProviderDetails {
  available?: boolean;
  enabled?: boolean;
  configured?: boolean;
  location?: string;
  endpoint_id?: string;
  error?: string;
  [key: string]: unknown;
}

export interface StatusResponse {
  product: string;
  name: string;
  model_status: string;
  trained_model_connected: boolean;
  model_metrics: ModelMetrics;
  providers: {
    active: string;
    local?: ProviderDetails;
    vertex?: ProviderDetails;
    error?: string;
  };
  copilot: {
    configured: boolean;
    provider: string;
    model: string | null;
    message: string;
  };
  input_schema: {
    numeric_features: NumericParameterName[];
    categorical_features: Array<"CarRear" | "Wheels">;
  };
  dataset: DatasetSummary;
}

export interface UncertaintyEstimate {
  estimate: number;
  lower: number;
  upper: number;
  basis: string;
}

export interface PredictionResponse {
  cd: number;
  percentile: number;
  level: "low" | "medium" | "high";
  comparison: string;
  provider: string;
  domain_status: DomainStatus;
  nearest_sample_distance: number;
  uncertainty: UncertaintyEstimate | null;
  warnings: string[];
  model: {
    name: string;
    status: string;
    confidence: string;
    metrics?: ModelMetrics;
    features_used?: number;
    message?: string;
  };
  dataset: DatasetSummary;
}

export interface SensitivityDriver {
  name: NumericParameterName;
  label: string;
  minus_value: number;
  plus_value: number;
  minus_cd: number;
  plus_cd: number;
  minus_delta: number;
  plus_delta: number;
  impact: number;
}

export interface AnalysisResponse {
  base_cd: number;
  provider: string;
  drivers: SensitivityDriver[];
  warnings: string[];
}

export interface RecommendationChange {
  name: NumericParameterName;
  label: string;
  from: number;
  to: number;
}

export interface Recommendation {
  cd: number;
  improvement: number;
  distance: number;
  domain_status: DomainStatus;
  parameters: DesignParameters;
  changes: RecommendationChange[];
}

export interface OptimizationRequest {
  parameters: DesignParameters;
  target_cd: number;
  locked: string[];
}

export interface OptimizationResponse {
  current_cd: number;
  target_cd: number;
  locked: string[];
  recommendations: Recommendation[];
  message: string;
}

export interface CopilotHistoryItem {
  role: "user" | "assistant";
  content: string;
}

export interface CopilotRequest {
  message: string;
  parameters: DesignParameters;
  history?: CopilotHistoryItem[];
}

export interface CopilotEvidence {
  prediction: Pick<
    PredictionResponse,
    | "cd"
    | "percentile"
    | "provider"
    | "domain_status"
    | "nearest_sample_distance"
    | "uncertainty"
    | "warnings"
  >;
  top_drivers: SensitivityDriver[];
  goal_search: OptimizationResponse | null;
}

export interface CopilotResponse {
  answer: string;
  provider: string;
  model: string | null;
  evidence: CopilotEvidence;
  disclaimer: string;
}

export interface VertexTestResponse {
  prediction: number;
  latency_ms: number;
  provider?: string;
  [key: string]: unknown;
}

export interface StlPredictionResponse {
  cd: number;
  percentile: number;
  level: "low" | "medium" | "high";
  comparison: string;
  model: PredictionResponse["model"];
  dataset: DatasetSummary;
  preview_points: [number, number, number][];
  file: { name: string; size_bytes: number };
  mesh: Record<string, unknown>;
}

export interface Variant {
  id: string;
  name: string;
  design: DesignParameters;
  cd: number | null;
  baselineDelta: number | null;
  thumbnail: string;
  savedAt: string;
}

export interface Baseline {
  design: DesignParameters;
  cd: number | null;
}

export interface RecommendationPreview {
  original: DesignParameters;
  recommendation: Recommendation;
}

export interface ApiErrorPayload {
  code?: string;
  message?: string;
  fallback_provider?: string;
  [key: string]: unknown;
}

export interface PointNetDemoItem {
  id: string;
  body_type: string;
  true_cd: number | null;
  /** 분포 밖으로 판단되면 null — 그때는 raw_cd만 참고용으로 노출한다. */
  cd: number | null;
  raw_cd: number;
  trusted: boolean;
  warnings: string[];
  error_counts?: number;
}

export interface PointNetDemoResponse {
  available: boolean;
  reason?: string;
  point_count?: number;
  mean_error_counts?: number | null;
  items: PointNetDemoItem[];
  note?: string;
}
