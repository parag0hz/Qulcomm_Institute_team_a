interface HeaderProps {
  designName: string;
  modelLabel: string;
  canUndo: boolean;
  canRedo: boolean;
  canSaveVariant: boolean;
  onNameChange: (name: string) => void;
  onUndo: () => void;
  onRedo: () => void;
  onSaveVariant: () => void;
  onReset: () => void;
  onExport: () => void;
  onOpenBenchmark: () => void;
}

export function Header({
  designName,
  modelLabel,
  canUndo,
  canRedo,
  canSaveVariant,
  onNameChange,
  onUndo,
  onRedo,
  onSaveVariant,
  onReset,
  onExport,
  onOpenBenchmark,
}: HeaderProps) {
  return (
    <header className="topbar">
      <a className="brand" href="/">
        <span className="brand-mark">P</span>
        <span><strong>PARAGON</strong><small>Vehicle design studio</small></span>
      </a>
      <div className="design-title">
        <span className="status-dot" />
        <input value={designName} onChange={(event) => onNameChange(event.target.value)} aria-label="Design name" />
        <span>Saved locally</span>
      </div>
      <div className="top-actions">
        <span className="model-chip">{modelLabel}</span>
        <button className="icon-button" type="button" title="Undo" disabled={!canUndo} onClick={onUndo}>↶</button>
        <button className="icon-button" type="button" title="Redo" disabled={!canRedo} onClick={onRedo}>↷</button>
        <button id="saveVariantHeader" className="ghost-button" type="button" disabled={!canSaveVariant} onClick={onSaveVariant}>Save variant</button>
        <button className="ghost-button" type="button" onClick={onOpenBenchmark}>Held-out benchmark</button>
        <button className="ghost-button topbar-reset" type="button" onClick={onReset}>Reset current</button>
        <button className="dark-button topbar-export" type="button" onClick={onExport}>Export design</button>
      </div>
    </header>
  );
}
