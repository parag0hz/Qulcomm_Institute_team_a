import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { DemoPage } from "./DemoPage";

const root = document.getElementById("root");
if (!root) throw new Error("Paragon root element is missing.");

// 라우터 의존성 없이 경로만 나눈다. FastAPI가 모든 경로에 index.html을 주므로
// 어느 경로로 들어와도 동작하고, 두 화면은 서로의 스타일시트를 물려받지 않는다.
// 정문(/)은 설명이 있는 데모다. 스튜디오는 '직접 해보기'로 들어가는 안쪽 화면.
const path = window.location.pathname.replace(/\/+$/, "");
const isStudio = path === "/studio";

createRoot(root).render(
  <StrictMode>{isStudio ? <App /> : <DemoPage />}</StrictMode>,
);
