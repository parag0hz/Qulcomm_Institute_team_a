import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { DemoPage } from "./DemoPage";

const root = document.getElementById("root");
if (!root) throw new Error("Paragon root element is missing.");

// 라우터 의존성 없이 경로만 나눈다. FastAPI가 모든 경로에 index.html을 주므로
// /demo 로 직접 들어와도 동작하고, 서로의 스타일시트를 물려받지 않는다.
const isDemo = window.location.pathname.replace(/\/+$/, "") === "/demo";

createRoot(root).render(
  <StrictMode>{isDemo ? <DemoPage /> : <App />}</StrictMode>,
);
