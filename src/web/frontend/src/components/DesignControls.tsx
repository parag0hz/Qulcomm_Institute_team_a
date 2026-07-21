import type { ChangeEvent } from "react";
import type { DesignParameters, ParameterDefinition, ParameterSchema } from "../types";

interface DesignControlsProps {
  schema: ParameterSchema;
  design: DesignParameters;
  activeParameter: string;
  locked: string[];
  mode: "parameters" | "stl";
  stlBusy: boolean;
  stlSummary?: string;
  onModeChange: (mode: "parameters" | "stl") => void;
  onParameterChange: (name: string, value: number) => void;
  onCategoryChange: (name: "CarRear" | "Wheels", value: string) => void;
  onPreset: (id: string) => void;
  onFocus: (name: string) => void;
  onToggleLock: (name: string) => void;
  onStlUpload: (file: File) => void;
}

const format = (value: number) => Number(value).toLocaleString("en-US", { maximumFractionDigits: 2 });

function ParameterControl({
  parameter,
  value,
  active,
  locked,
  onChange,
  onFocus,
  onToggleLock,
}: {
  parameter: ParameterDefinition;
  value: number;
  active: boolean;
  locked: boolean;
  onChange: (value: number) => void;
  onFocus: () => void;
  onToggleLock: () => void;
}) {
  return (
    <div className={`parameter-control ${active ? "active" : ""}`} data-control={parameter.name} onFocus={onFocus}>
      <div className="parameter-head">
        <label htmlFor={`range-${parameter.name}`}>{parameter.label}</label>
        <button className={`lock-control ${locked ? "active" : ""}`} type="button" title="Lock for optimization" aria-pressed={locked} onClick={onToggleLock}>{locked ? "●" : "○"}</button>
        <input
          className="parameter-number"
          type="number"
          value={value}
          min={parameter.min}
          max={parameter.max}
          step={parameter.step}
          aria-label={`${parameter.label} value`}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </div>
      <div className="slider-row">
        <input
          id={`range-${parameter.name}`}
          type="range"
          value={value}
          min={parameter.min}
          max={parameter.max}
          step={parameter.step}
          onInput={(event) => onChange(Number((event.target as HTMLInputElement).value))}
          onFocus={onFocus}
        />
        <div className="range-hints"><span>{format(parameter.min)}</span><span>{format(parameter.max)}</span></div>
      </div>
    </div>
  );
}

export function DesignControls(props: DesignControlsProps) {
  const { schema, design, activeParameter, locked } = props;
  const primary = schema.parameters.filter((item) => item.high_impact);
  const secondary = schema.parameters.filter((item) => !item.high_impact);
  const groups = secondary.reduce<Record<string, ParameterDefinition[]>>((accumulator, item) => {
    (accumulator[item.group] ||= []).push(item);
    return accumulator;
  }, {});
  const validCombination = schema.valid_combinations.some((item) => item.CarRear === design.CarRear && item.Wheels === design.Wheels);
  const renderControl = (parameter: ParameterDefinition) => (
    <ParameterControl
      key={parameter.name}
      parameter={parameter}
      value={Number(design[parameter.name])}
      active={activeParameter === parameter.name}
      locked={locked.includes(parameter.name)}
      onChange={(value) => props.onParameterChange(parameter.name, value)}
      onFocus={() => props.onFocus(parameter.name)}
      onToggleLock={() => props.onToggleLock(parameter.name)}
    />
  );

  const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) props.onStlUpload(file);
  };

  return (
    <aside className="control-panel">
      <div className="panel-intro">
        <p className="eyebrow">Design inputs</p>
        <h1>Explore your concept</h1>
        <p>Adjust model inputs to explore approximate geometry changes and update the predicted Cd.</p>
      </div>
      <div className="mode-tabs" role="tablist">
        {(["parameters", "stl"] as const).map((mode) => (
          <button key={mode} className={`mode-tab ${props.mode === mode ? "active" : ""}`} type="button" onClick={() => props.onModeChange(mode)}>{mode === "parameters" ? "Parameters" : "Import STL"}</button>
        ))}
      </div>

      {props.mode === "parameters" ? (
        <section className="mode-panel">
          <label className="preset-field">Design preset
            <select defaultValue="" onChange={(event) => props.onPreset(event.target.value)}>
              <option value="">Choose a starting point</option>
              {schema.presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
            </select>
          </label>
          <div className="category-grid">
            <label>Body architecture
              <select value={design.CarRear} onChange={(event) => props.onCategoryChange("CarRear", event.target.value)}>{schema.categories.CarRear.map((value) => <option key={value}>{value}</option>)}</select>
            </label>
            <label>Wheel treatment
              <select value={design.Wheels} onChange={(event) => props.onCategoryChange("Wheels", event.target.value)}>{schema.categories.Wheels.map((value) => <option key={value}>{value}</option>)}</select>
            </label>
          </div>
          {!validCombination && <div className="combination-warning">This body and wheel combination was not present in training. Treat the estimate as out-of-domain.</div>}
          <div className="geometry-sync-note"><strong>Live concept geometry</strong><span>Approximate geometry morph · Cd model connected. Dimensions show changes from the reference mesh.</span></div>
          <div className="section-heading"><div><span className="section-kicker">High impact</span><h2>Aerodynamic drivers</h2></div><span className="impact-pill">EDA</span></div>
          <div className="control-list">{primary.map(renderControl)}</div>
          <details className="advanced-controls">
            <summary><span>All design parameters</span><small>{secondary.length} additional controls</small></summary>
            <div id="parameterGroups">
              {Object.entries(groups).map(([group, parameters]) => <section className="parameter-group" key={group}><h3>{group}</h3>{parameters.map(renderControl)}</section>)}
            </div>
          </details>
        </section>
      ) : (
        <section className="mode-panel">
          <label className="dropzone">
            <input type="file" accept=".stl,model/stl" disabled={props.stlBusy} onChange={handleFile} />
            <span className="dropzone-mark">STL</span>
            <strong>{props.stlBusy ? "Analyzing mesh…" : "Choose your vehicle mesh"}</strong>
            <small>ASCII or binary · up to 32 MB</small>
          </label>
          {props.stlSummary && <div className="stl-summary">{props.stlSummary}</div>}
          <div className="mode-note"><strong>Different prediction path</strong><p>STL analysis currently uses geometric proportions and is lower confidence than the trained parametric model.</p></div>
        </section>
      )}
    </aside>
  );
}
