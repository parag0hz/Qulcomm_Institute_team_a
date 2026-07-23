"""Prediction and dataset comparison logic for Paragon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import pickle
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .providers import ProviderRouter
from .stl import Point, normalize_preview_points


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PARAMETRIC_DATA_PATH = BACKEND_ROOT / "ParametricModels" / "DrivAerNet_ParametricData.csv"
DEFAULT_MODEL_PATH = BACKEND_ROOT / "artifacts" / "cfa_parametric_baseline.pkl"
TARGET_COLUMN = "Average Cd"
NON_FEATURE_COLUMNS = {
    "Experiment",
    "Design",
    "Average Cd",
    "Std Cd",
    "Average Cl",
    "Std Cl",
    "Average Cl_f",
    "Std Cl_f",
    "Average Cl_r",
    "Std Cl_r",
}

PARAMETER_LABELS = {
    "B_Ramp_Angle": "Ramp angle",
    "B_Diffusor_Angle": "Diffuser angle",
    "B_Trunklid_Angle": "Trunklid angle",
    "C_Side_Mirrors_Rotation": "Mirror rotation",
    "D_Rear_Window_Inclination": "Rear window inclination",
    "D_Winscreen_Inclination": "Windscreen inclination",
    "C_Side_Mirrors_Translate_X": "Mirror position X",
    "C_Side_Mirrors_Translate_Z": "Mirror position Z",
    "D_Winscreen_Length": "Windscreen length",
    "D_Rear_Window_Length": "Rear window length",
    "E_A_B_C_Pillar_Thickness": "A/B/C pillar thickness",
    "G_Trunklid_Curvature": "Trunklid curvature",
    "G_Trunklid_Length": "Trunklid length",
    "H_Front_Bumper_Curvature": "Front bumper curvature",
    "H_Front_Bumper_Length": "Front bumper length",
    "F_Door_Handles_Thickness": "Door handle thickness",
    "F_Door_Handles_Z_Position": "Door handle position Z",
    "E_Fenders_Arch_Offset": "Fender arch offset",
    "A_Car_Length": "Car length",
    "F_Door_Handles_X_Position": "Door handle position X",
    "A_Car_Width": "Car width",
    "A_Car_Roof_Height": "Roof height",
    "A_Car_Green_House_Angle": "Greenhouse angle",
}

PARAMETER_GROUPS = {
    "A_": "Body proportions",
    "B_": "Rear aerodynamics",
    "C_": "Mirrors",
    "D_": "Glasshouse",
    "E_": "Body details",
    "F_": "Door details",
    "G_": "Trunklid",
    "H_": "Front bumper",
}

HIGH_IMPACT_PARAMETERS = {
    "E_Fenders_Arch_Offset",
    "B_Diffusor_Angle",
    "E_A_B_C_Pillar_Thickness",
}

_PROVIDER_ROUTER: ProviderRouter | None = None
_DESIGN_ROWS: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class DatasetStats:
    count: int
    cd_min: float
    cd_max: float
    cd_mean: float
    cd_median: float
    cd_p25: float
    cd_p75: float
    cd_values: Tuple[float, ...]
    feature_columns: Tuple[str, ...]

    def public_dict(self) -> Dict[str, object]:
        return {
            "sample_count": self.count,
            "cd_min": round(self.cd_min, 4),
            "cd_max": round(self.cd_max, 4),
            "cd_mean": round(self.cd_mean, 4),
            "cd_median": round(self.cd_median, 4),
            "cd_p25": round(self.cd_p25, 4),
            "cd_p75": round(self.cd_p75, 4),
            "feature_count": len(self.feature_columns),
            "feature_columns": list(self.feature_columns),
        }


def load_dataset_stats(csv_path: Path = PARAMETRIC_DATA_PATH) -> DatasetStats:
    if not csv_path.exists():
        return _empty_stats()

    cd_values: List[float] = []
    feature_columns: Tuple[str, ...] = ()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            feature_columns = tuple(
                column for column in reader.fieldnames if column not in NON_FEATURE_COLUMNS
            )
        for row in reader:
            value = _safe_float(row.get(TARGET_COLUMN))
            if value is not None:
                cd_values.append(value)

    if not cd_values:
        return _empty_stats(feature_columns=feature_columns)

    cd_values.sort()
    return DatasetStats(
        count=len(cd_values),
        cd_min=cd_values[0],
        cd_max=cd_values[-1],
        cd_mean=statistics.fmean(cd_values),
        cd_median=statistics.median(cd_values),
        cd_p25=_quantile(cd_values, 0.25),
        cd_p75=_quantile(cd_values, 0.75),
        cd_values=tuple(cd_values),
        feature_columns=feature_columns,
    )


def load_parameter_schema(csv_path: Path = PARAMETRIC_DATA_PATH) -> Dict[str, object]:
    """Return design controls and supported category combinations for the UI."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Parametric dataset not found: {csv_path}")

    values: Dict[str, List[float]] = {}
    combinations = set()
    rear_names = {"F": "Fastback", "E": "Estateback", "N": "Notchback"}
    wheel_names = {"WW": "Open detailed", "WWS": "Open smooth", "WWC": "Closed smooth"}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            values = {
                column: []
                for column in reader.fieldnames
                if column not in NON_FEATURE_COLUMNS
            }
        for row in reader:
            tokens = row.get("Experiment", "").split("_")
            if len(tokens) >= 3 and tokens[0] in rear_names and tokens[2] in wheel_names:
                combinations.add((rear_names[tokens[0]], wheel_names[tokens[2]]))
            for column in values:
                value = _safe_float(row.get(column))
                if value is not None:
                    values[column].append(value)

    parameters = []
    for column, column_values in values.items():
        if not column_values:
            continue
        column_values.sort()
        prefix = column[:2]
        parameters.append(
            {
                "name": column,
                "label": PARAMETER_LABELS.get(column, column.replace("_", " ")),
                "group": PARAMETER_GROUPS.get(prefix, "Other"),
                "min": round(column_values[0], 5),
                "max": round(column_values[-1], 5),
                "default": round(statistics.median(column_values), 5),
                "step": round(max((column_values[-1] - column_values[0]) / 200, 0.01), 5),
                "high_impact": column in HIGH_IMPACT_PARAMETERS,
            }
        )

    schema = {
        "parameters": parameters,
        "categories": {
            "CarRear": list(rear_names.values()),
            "Wheels": list(wheel_names.values()),
        },
        "valid_combinations": [
            {"CarRear": rear, "Wheels": wheels}
            for rear, wheels in sorted(combinations)
        ],
    }
    schema["presets"] = _build_presets(csv_path, parameters)
    _apply_in_domain_defaults(schema)
    return schema


def _apply_in_domain_defaults(schema: Dict[str, object]) -> None:
    """Seed the opening design from a real observed sample.

    Each parameter's median is a sensible per-column default, but combining 23 of
    them lands on a point no sample actually occupies, so the studio's very first
    prediction reports an `outside` domain and shows a warning before the user has
    touched anything. Reusing the representative Fastback row keeps that first
    screen inside the observed domain. The presets are built beforehand, so
    "Dataset median" keeps its original per-column meaning.
    """

    presets = schema.get("presets") or []
    seed = next((preset for preset in presets if preset.get("id") == "fastback"), None)
    if seed is None:
        return

    design = seed.get("design") or {}
    for parameter in schema["parameters"]:
        value = _safe_float(design.get(parameter["name"]))
        if value is None:
            continue
        bounded = min(parameter["max"], max(parameter["min"], value))
        parameter["default"] = round(bounded, 5)


def model_status(model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, object]:
    if not model_path.exists():
        return {"connected": False, "name": "not trained", "metrics": {}}
    try:
        with model_path.open("rb") as handle:
            artifact = pickle.load(handle)
        return {
            "connected": True,
            "name": artifact.get("model_name", "parametric baseline"),
            "metrics": artifact.get("metrics", {}),
        }
    except Exception:
        return {"connected": False, "name": "artifact error", "metrics": {}}


def provider_status(model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, object]:
    try:
        return _router(model_path).status()
    except Exception as exc:
        return {"active": "Unavailable", "error": str(exc), "local": {"available": False}, "vertex": {"available": False}}


def test_vertex_provider(parameters: Mapping[str, object] | None = None) -> Dict[str, object]:
    router = _router(DEFAULT_MODEL_PATH)
    if parameters is None:
        schema = load_parameter_schema()
        parameters = {item["name"]: item["default"] for item in schema["parameters"]}
        parameters = {**parameters, "CarRear": "Fastback", "Wheels": "Closed smooth"}
    row = _validated_row(parameters)
    return router.vertex.test(row, router.local.artifact.get("numeric_columns", []))


def predict_from_stl_points(
    points: Sequence[Point],
    triangle_count: int,
    source_format: str,
    stats: DatasetStats | None = None,
) -> Dict[str, object]:
    """Predict Cd from uploaded STL geometry.

    This MVP intentionally marks the result as a geometry fallback. It is useful
    for proving the upload-to-result product flow, but it is not a trained CFD
    surrogate.
    """

    stats = stats or load_dataset_stats()
    geometry = summarize_geometry(points)
    cd = _fallback_cd_from_geometry(geometry, stats)
    percentile = percentile_rank(cd, stats.cd_values)
    level = drag_level(percentile)

    return {
        "cd": round(cd, 4),
        "percentile": round(percentile, 1),
        "level": level,
        "comparison": comparison_label(percentile),
        "model": {
            "name": "geometry_fallback",
            "status": "not_trained",
            "confidence": "low",
            "message": "Trained LightGBM/XGBoost or PointNet artifact is not connected yet.",
        },
        "dataset": stats.public_dict(),
        "mesh": {
            "triangle_count": triangle_count,
            "point_count": len(points),
            "source_format": source_format,
            **geometry,
        },
        "preview_points": normalize_preview_points(points),
    }


def summarize_geometry(points: Sequence[Point]) -> Dict[str, object]:
    if not points:
        raise ValueError("Point cloud has no vertices.")

    xs, ys, zs = zip(*points)
    extents = {
        "x": max(xs) - min(xs),
        "y": max(ys) - min(ys),
        "z": max(zs) - min(zs),
    }
    ordered = sorted(extents.values(), reverse=True)
    length = max(ordered[0], 1e-9)
    width = max(ordered[1] if len(ordered) > 1 else 0.0, 1e-9)
    height = max(ordered[2] if len(ordered) > 2 else 0.0, 1e-9)
    frontal_ratio = (width * height) / (length * length)
    slenderness = length / max(width, height, 1e-9)

    return {
        "bounds": {
            "x": [round(min(xs), 4), round(max(xs), 4)],
            "y": [round(min(ys), 4), round(max(ys), 4)],
            "z": [round(min(zs), 4), round(max(zs), 4)],
        },
        "extents": {axis: round(value, 4) for axis, value in extents.items()},
        "normalized_shape": {
            "length": round(length, 4),
            "width": round(width, 4),
            "height": round(height, 4),
            "frontal_ratio": round(frontal_ratio, 4),
            "slenderness": round(slenderness, 4),
        },
    }


def maybe_predict_parameters(
    parameters: Mapping[str, object],
    model_path: Path = DEFAULT_MODEL_PATH,
    stats: DatasetStats | None = None,
) -> Dict[str, object]:
    """Predict Cd from parametric inputs when a trained artifact exists."""

    stats = stats or load_dataset_stats()
    router = _router(model_path)
    artifact = router.local.artifact

    feature_columns = artifact["feature_columns"]
    missing = [column for column in feature_columns if column not in parameters]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    numeric_columns = artifact.get("numeric_columns", feature_columns)
    categorical_columns = artifact.get("categorical_columns", [])
    row = {
        column: (
            str(parameters[column])
            if column in categorical_columns
            else float(parameters[column])
        )
        for column in feature_columns
    }
    prediction = router.predict([row])
    cd = prediction.values[0]
    percentile = percentile_rank(cd, stats.cd_values)
    domain = analyze_domain(row)
    mae = float(artifact.get("metrics", {}).get("mae", 0.0))
    std_cd = float(domain.get("nearest_std_cd", 0.0))
    uncertainty = math.sqrt(mae * mae + std_cd * std_cd)
    warnings = [*prediction.warnings, *domain["warnings"]]

    return {
        "cd": round(cd, 4),
        "percentile": round(percentile, 1),
        "level": drag_level(percentile),
        "comparison": comparison_label(percentile),
        "provider": prediction.provider,
        "domain_status": domain["status"],
        "nearest_sample_distance": domain["distance"],
        "uncertainty": {
            "estimate": round(uncertainty, 4),
            "lower": round(cd - 1.96 * uncertainty, 4),
            "upper": round(cd + 1.96 * uncertainty, 4),
            "basis": "validation MAE combined with nearest-sample Std Cd",
        },
        "warnings": warnings,
        "model": {
            "name": artifact.get("model_name", "parametric_baseline"),
            "status": "trained",
            "confidence": "medium",
            "metrics": artifact.get("metrics", {}),
            "features_used": len(numeric_columns) + len(categorical_columns),
        },
        "dataset": stats.public_dict(),
    }


def analyze_parameters(parameters: Mapping[str, object]) -> Dict[str, object]:
    schema = load_parameter_schema()
    base = _validated_row(parameters)
    candidates = []
    metadata = []
    for parameter in schema["parameters"]:
        name = parameter["name"]
        amount = max(float(parameter["step"]), (float(parameter["max"]) - float(parameter["min"])) * 0.05)
        minus = dict(base)
        plus = dict(base)
        minus[name] = max(float(parameter["min"]), float(base[name]) - amount)
        plus[name] = min(float(parameter["max"]), float(base[name]) + amount)
        candidates.extend([minus, plus])
        metadata.append((parameter, minus[name], plus[name]))
    prediction = _router(DEFAULT_MODEL_PATH).predict(candidates)
    base_cd = maybe_predict_parameters(base)["cd"]
    drivers = []
    for index, (parameter, minus_value, plus_value) in enumerate(metadata):
        minus_cd, plus_cd = prediction.values[index * 2:index * 2 + 2]
        drivers.append({
            "name": parameter["name"], "label": parameter["label"],
            "minus_value": round(float(minus_value), 5), "plus_value": round(float(plus_value), 5),
            "minus_cd": round(minus_cd, 4), "plus_cd": round(plus_cd, 4),
            "minus_delta": round(minus_cd - base_cd, 5), "plus_delta": round(plus_cd - base_cd, 5),
            "impact": round(max(abs(minus_cd - base_cd), abs(plus_cd - base_cd)), 5),
        })
    drivers.sort(key=lambda item: item["impact"], reverse=True)
    return {"base_cd": base_cd, "provider": prediction.provider, "drivers": drivers, "warnings": prediction.warnings}


def optimize_parameters(parameters: Mapping[str, object], target_cd: float, locked: Sequence[str] = ()) -> Dict[str, object]:
    schema = load_parameter_schema()
    current = _validated_row(parameters)
    locked_set = set(locked) | {"CarRear", "Wheels"}
    current_cd = maybe_predict_parameters(current)["cd"]
    pool = []
    working = dict(current)
    seen = set()
    for _ in range(4):
        candidates = []
        for parameter in schema["parameters"]:
            name = parameter["name"]
            if name in locked_set:
                continue
            amount = max(float(parameter["step"]), (float(parameter["max"]) - float(parameter["min"])) * 0.05)
            for direction in (-1, 1):
                candidate = dict(working)
                candidate[name] = min(float(parameter["max"]), max(float(parameter["min"]), float(working[name]) + direction * amount))
                key = tuple(round(float(candidate[item["name"]]), 6) for item in schema["parameters"])
                if key not in seen:
                    seen.add(key); candidates.append(candidate)
        if not candidates:
            break
        result = _router(DEFAULT_MODEL_PATH).predict(candidates)
        ranked = sorted(zip(candidates, result.values), key=lambda item: (abs(item[1] - target_cd), item[1]))
        for candidate, cd in ranked[:5]:
            pool.append((candidate, cd))
        working = dict(ranked[0][0])
    pool.sort(key=lambda item: (abs(item[1] - target_cd), item[1]))
    recommendations = []
    used = set()
    labels = {item["name"]: item["label"] for item in schema["parameters"]}
    for candidate, cd in pool:
        key = tuple(round(float(candidate[item["name"]]), 6) for item in schema["parameters"])
        if key in used:
            continue
        used.add(key)
        changes = [{"name": name, "label": labels[name], "from": round(float(current[name]), 5), "to": round(float(candidate[name]), 5)}
                   for name in labels if abs(float(candidate[name]) - float(current[name])) > 1e-9]
        domain = analyze_domain(candidate)
        recommendations.append({"cd": round(cd, 4), "improvement": round(current_cd - cd, 4),
                                "distance": domain["distance"], "domain_status": domain["status"],
                                "parameters": candidate, "changes": changes})
        if len(recommendations) == 3:
            break
    return {"current_cd": current_cd, "target_cd": round(float(target_cd), 4), "locked": sorted(locked_set), "recommendations": recommendations,
            "message": "Surrogate-model design guidance; validate shortlisted concepts with CFD."}


def analyze_domain(row: Mapping[str, object]) -> Dict[str, object]:
    rows = _load_design_rows()
    schema = load_parameter_schema()
    numeric = [item["name"] for item in schema["parameters"]]
    matching = [item for item in rows if item["CarRear"] == row.get("CarRear") and item["Wheels"] == row.get("Wheels")]
    warnings = []
    if not matching:
        matching = rows
        warnings.append("This body and wheel combination was not present in training.")
    def distance(item):
        parts = []
        for parameter in schema["parameters"]:
            span = max(float(parameter["max"]) - float(parameter["min"]), 1e-9)
            parts.append(((float(row[parameter["name"]]) - float(item[parameter["name"]])) / span) ** 2)
        return math.sqrt(statistics.fmean(parts))
    nearest = min(matching, key=distance)
    nearest_distance = distance(nearest)
    outside_range = [name for name in numeric if float(row[name]) < next(p["min"] for p in schema["parameters"] if p["name"] == name)
                     or float(row[name]) > next(p["max"] for p in schema["parameters"] if p["name"] == name)]
    status = "outside" if outside_range or nearest_distance > 0.25 or not [item for item in rows if item["CarRear"] == row.get("CarRear") and item["Wheels"] == row.get("Wheels")] else "edge" if nearest_distance > 0.12 else "inside"
    if status == "edge": warnings.append("Inputs are near the edge of the observed training domain.")
    if status == "outside": warnings.append("Prediction is outside or far from the observed training domain.")
    std_cd = float(nearest.get("Std Cd", 0.0))
    if std_cd > 0.012: warnings.append(f"Nearest sample has high CFD variation (Std Cd {std_cd:.4f}).")
    return {"status": status, "distance": round(nearest_distance, 4), "nearest_experiment": nearest.get("Experiment"),
            "nearest_std_cd": round(std_cd, 4), "warnings": warnings}


def _router(model_path: Path) -> ProviderRouter:
    global _PROVIDER_ROUTER
    if _PROVIDER_ROUTER is None or _PROVIDER_ROUTER.local.model_path != model_path:
        _PROVIDER_ROUTER = ProviderRouter(model_path)
    return _PROVIDER_ROUTER


def _validated_row(parameters: Mapping[str, object]) -> dict[str, object]:
    artifact = _router(DEFAULT_MODEL_PATH).local.artifact
    missing = [name for name in artifact["feature_columns"] if name not in parameters]
    if missing: raise ValueError(f"Missing required parameters: {', '.join(missing)}")
    return {name: str(parameters[name]) if name in artifact.get("categorical_columns", []) else float(parameters[name])
            for name in artifact["feature_columns"]}


def _load_design_rows() -> list[dict[str, object]]:
    global _DESIGN_ROWS
    if _DESIGN_ROWS is not None: return _DESIGN_ROWS
    rear_names = {"F": "Fastback", "E": "Estateback", "N": "Notchback"}
    wheel_names = {"WW": "Open detailed", "WWS": "Open smooth", "WWC": "Closed smooth"}
    rows = []
    with PARAMETRIC_DATA_PATH.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            tokens = source["Experiment"].split("_")
            row = {name: float(value) for name, value in source.items() if name not in NON_FEATURE_COLUMNS}
            row.update({"Experiment": source["Experiment"], "Std Cd": float(source.get("Std Cd") or 0),
                        "Average Cd": float(source[TARGET_COLUMN]), "CarRear": rear_names[tokens[0]], "Wheels": wheel_names[tokens[2]]})
            rows.append(row)
    _DESIGN_ROWS = rows
    return rows


def _build_presets(csv_path: Path, parameters: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = _load_design_rows() if csv_path == PARAMETRIC_DATA_PATH else []
    defaults = {item["name"]: item["default"] for item in parameters}
    def design(row): return {**{item["name"]: row[item["name"]] for item in parameters}, "CarRear": row["CarRear"], "Wheels": row["Wheels"]}
    presets = [{"id": "median", "name": "Dataset median", "design": {**defaults, "CarRear": "Fastback", "Wheels": "Closed smooth"}},
               {"id": "reference", "name": "Current reference geometry", "design": {**defaults, "CarRear": "Fastback", "Wheels": "Open detailed"}}]
    if rows:
        presets.insert(1, {"id": "lowest", "name": "Lowest-Cd observed sample", "design": design(min(rows, key=lambda item: item["Average Cd"]))})
        fastback = [row for row in rows if row["CarRear"] == "Fastback"]
        presets.insert(2, {"id": "fastback", "name": "Fastback baseline", "design": design(min(fastback, key=lambda item: abs(item["Average Cd"] - statistics.median(r["Average Cd"] for r in fastback))))})
    return presets


def percentile_rank(value: float, sorted_values: Sequence[float]) -> float:
    if not sorted_values:
        return 50.0
    below_or_equal = sum(1 for item in sorted_values if item <= value)
    return below_or_equal / len(sorted_values) * 100.0


def drag_level(percentile: float) -> str:
    if percentile <= 35:
        return "low"
    if percentile <= 70:
        return "medium"
    return "high"


def comparison_label(percentile: float) -> str:
    if percentile <= 35:
        return "Better than most sample designs"
    if percentile <= 70:
        return "Near the sample median"
    return "Higher drag than most sample designs"


def _fallback_cd_from_geometry(geometry: Mapping[str, object], stats: DatasetStats) -> float:
    shape = geometry["normalized_shape"]
    assert isinstance(shape, Mapping)
    frontal_ratio = float(shape["frontal_ratio"])
    slenderness = float(shape["slenderness"])

    # Lower frontal ratio and higher slenderness generally reduce drag. The
    # coefficient is calibrated only to the local DrivAerNet Cd range.
    drag_index = 0.68 * frontal_ratio + 0.32 * (1.0 / max(slenderness, 1e-9))
    centered = drag_index - 0.22
    cd = stats.cd_mean + centered * 0.18
    return min(max(cd, stats.cd_min), stats.cd_max)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_stats(feature_columns: Iterable[str] = ()) -> DatasetStats:
    values = (0.24, 0.27, 0.31)
    return DatasetStats(
        count=0,
        cd_min=values[0],
        cd_max=values[-1],
        cd_mean=statistics.fmean(values),
        cd_median=statistics.median(values),
        cd_p25=_quantile(values, 0.25),
        cd_p75=_quantile(values, 0.75),
        cd_values=values,
        feature_columns=tuple(feature_columns),
    )
