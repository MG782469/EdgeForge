// EdgeForge Studio - builds studio/web/dist (React + TypeScript, bundled with esbuild).
//   cd studio/web && npm install && node build.mjs        (dist/ is committed: rebuilding is optional)
import { build } from "esbuild";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, "dist");
fs.mkdirSync(dist, { recursive: true });
await build({
  entryPoints: [path.join(here, "src", "main.tsx")],
  bundle: true,
  minify: true,
  format: "iife",
  target: "es2020",
  jsx: "automatic",
  outfile: path.join(dist, "app.js"),
  define: { "process.env.NODE_ENV": '"production"' },
  nodePaths: process.env.NODE_PATH ? process.env.NODE_PATH.split(path.delimiter) : [],
  logLevel: "info",
});
fs.copyFileSync(path.join(here, "src", "styles.css"), path.join(dist, "app.css"));
fs.copyFileSync(path.join(here, "index.html"), path.join(dist, "index.html"));
console.log("built", dist);
