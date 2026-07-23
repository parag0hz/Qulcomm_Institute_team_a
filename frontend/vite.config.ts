import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const backend = "http://127.0.0.1:8001";
const frontendRoot = decodeURIComponent(new URL(".", import.meta.url).pathname);

export default defineConfig({
  root: frontendRoot,
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": backend,
      "/static/models": backend,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    // Three.js is intentionally lazy-loaded as the dedicated viewer chunk.
    chunkSizeWarningLimit: 650,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    clearMocks: true,
    restoreMocks: true,
  },
});
