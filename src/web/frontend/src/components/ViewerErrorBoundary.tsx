import { Component, type ErrorInfo, type ReactNode } from "react";

interface ViewerErrorBoundaryProps {
  children: ReactNode;
}

interface ViewerErrorBoundaryState {
  failed: boolean;
}

export class ViewerErrorBoundary extends Component<
  ViewerErrorBoundaryProps,
  ViewerErrorBoundaryState
> {
  state: ViewerErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ViewerErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("Paragon 3D viewer failed.", error, info);
  }

  render(): ReactNode {
    if (this.state.failed) {
      return (
        <div className="viewer-host viewer-unavailable" data-testid="vehicle-viewer-fallback">
          <div className="viewer-error" role="alert">
            <strong>3D preview unavailable</strong>
            <span>The design controls and prediction results are still available.</span>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
