#!/usr/bin/env node
// EdgeForge Studio - local bridge between the browser and the real EdgeForge C++ engine.
//
//   node studio/server.mjs --model-dir DIR --efq FILE [--bin ~/ef-build/edgeforge] [--port 8790] [--host 127.0.0.1]
//
// - Runs `edgeforge chat` (the unchanged CLI) as a child process and talks to it over stdin/stdout:
//   the browser's prompt goes in as one line, the streamed reply and the engine's own stats line
//   come back. Stop = kill the engine and start it again (the conversation is cleared).
// - Samples the engine process from Linux /proc every 100 ms: each thread's CPU time and the
//   (v)CPU it last ran on, process RSS, and per-logical-CPU load. These are OS measurements of
//   the real engine; nothing is simulated. Per-job scheduler events are not exported by this
//   build of the engine and are reported as "not instrumented".
// - Serves the built web app (studio/web/dist) and a WebSocket at /ws. Node built-ins only.
import { spawn, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const DIST = path.join(HERE, "web", "dist");

// ---------------------------------------------------------------- options
const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
};
const expand = (p) => (p ? p.replace(/^~(?=$|\/)/, os.homedir()) : p);
const CFG = {
  bin: expand(opt("bin", process.env.EF_BIN || "~/ef-build/edgeforge")),
  modelDir: expand(opt("model-dir", process.env.EF_MODEL_DIR || "~/edgeforge-data/tinyllama")),
  efq: expand(opt("efq", process.env.EF_EFQ || "~/edgeforge-data/tinyllama-q4.efq")),
  port: Number(opt("port", process.env.PORT || 8790)),
  host: opt("host", "127.0.0.1"),
};
const LOGICAL = os.cpus().length;
let CLK_TCK = 100;
try {
  CLK_TCK = Number(execFileSync("getconf", ["CLK_TCK"]).toString().trim()) || 100;
} catch {}

// ---------------------------------------------------------------- static facts (measured once)
function sysinfo() {
  try {
    const out = execFileSync(CFG.bin, ["sysinfo"], { cwd: ROOT, timeout: 10000 }).toString();
    const pick = (re) => (out.match(re) || [])[1] || "";
    return {
      cpu: pick(/CPU\s*:\s*(.+)/),
      cores: pick(/Cores\s*:\s*([^|]+)/).trim(),
      os: pick(/\| OS (.+)/),
      ram: pick(/RAM\s*:\s*(.+)/),
      features: pick(/Features\s*:\s*(.+)/),
      build: pick(/Build\s*:\s*(.+)/),
      physical: Number(pick(/(\d+) physical/)) || 0,
      logical: Number(pick(/(\d+) logical/)) || LOGICAL,
      wsl: /microsoft|WSL/i.test(out) || /microsoft/i.test(os.release()),
    };
  } catch (e) {
    return { error: String(e.message || e), logical: LOGICAL, wsl: /microsoft/i.test(os.release()) };
  }
}

function fileSize(p) {
  try {
    return fs.statSync(p).size;
  } catch {
    return null;
  }
}

// Historical numbers from files in the repository, with their source and date (never shown as live).
function historical() {
  const h = { membench: null, ladders: [], source: [] };
  try {
    const mb = JSON.parse(fs.readFileSync(path.join(ROOT, "bench", "membench.json"), "utf8"));
    h.membench = { gbps: mb.peak_median_gbps, threads: mb.peak_threads, file: "bench/membench.json",
                   date: mb.version || "", cpu: (mb.system && mb.system.cpu) || "" };
  } catch {}
  try {
    for (const f of fs.readdirSync(path.join(ROOT, "bench")).filter((x) => /^report_.*\.md$/.test(x))) {
      const text = fs.readFileSync(path.join(ROOT, "bench", f), "utf8");
      const date = (text.match(/date\s*:\s*(\S+)/) || [])[1] || "";
      for (const sec of text.split(/\n## /)) {
        const m = sec.match(/^Speed-up ladder, (.+?) \(bench-decode\)/);
        if (!m) continue;
        const rows = [];
        for (const line of sec.split("\n")) {
          const r = line.match(/^\s+(scalar|avx2)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)/);
          if (r) rows.push({ kernels: r[1], threads: +r[2], prefill: +r[3], decode: +r[4], gbps: +r[5] });
        }
        h.ladders.push({ engine: m[1], file: `bench/${f}`, date, rows });
      }
    }
  } catch {}
  return h;
}

// ---------------------------------------------------------------- engine process
const STATS_RE =
  /\n {2}\[prompt (\d+) tok, (\d+) reused, (\d+) computed in ([\d.]+) s \| reply (\d+) tok in ([\d.]+) s = ([\d.]+) tok\/s \| end: ([^\]]*)\]\n/;
const STATS_HEAD = "\n  [prompt ";

const clients = new Set();
function broadcast(msg) {
  const data = JSON.stringify(msg);
  for (const c of clients) c.send(data);
}

const engine = {
  proc: null,
  state: "stopped", // starting | ready | generating | stopped | error
  settings: { model: "q4", kernels: "auto", threads: 0, maxNew: 256, temp: 0, topP: 1, repPenalty: 1, eosMinP: 0,
              system: "default", customSystem: "" },
  banner: "",
  buf: "",
  run: null, // current generation
  history: [],
  lastError: "",
};

function engineArgs(s) {
  const a = ["chat"];
  if (s.model === "bf16") a.push("--model-dir", CFG.modelDir);
  else a.push("--qmodel", CFG.efq, "--tokenizer-dir", CFG.modelDir);
  if (s.kernels !== "auto") a.push("--kernels", s.kernels);
  if (s.threads > 0) a.push("--threads", String(s.threads));
  a.push("--max-new", String(s.maxNew));
  if (s.temp > 0) a.push("--temp", String(s.temp), "--top-p", String(s.topP), "--seed", "1");
  if (s.repPenalty !== 1) a.push("--rep-penalty", String(s.repPenalty));
  if (s.eosMinP > 0) a.push("--eos-min-p", String(s.eosMinP));
  if (s.system === "none") a.push("--no-system");
  if (s.system === "custom" && s.customSystem.trim()) a.push("--system", s.customSystem.trim());
  return a;
}

function setState(st, extra = {}) {
  engine.state = st;
  broadcast({ type: "engine", state: st, settings: engine.settings, banner: engine.banner, args: engine.args,
              pid: engine.proc ? engine.proc.pid : null, error: engine.lastError, ...extra });
}

function startEngine() {
  stopEngine();
  engine.args = engineArgs(engine.settings);
  engine.banner = "";
  engine.buf = "";
  engine.lastError = "";
  const p = spawn(CFG.bin, engine.args, { cwd: ROOT, stdio: ["pipe", "pipe", "pipe"] });
  engine.proc = p;
  engine.startedAt = Date.now();
  setState("starting");
  p.stdout.setEncoding("utf8");
  p.stdout.on("data", (chunk) => onEngineOutput(p, chunk));
  p.stderr.setEncoding("utf8");
  p.stderr.on("data", (d) => {
    engine.lastError += d;
    broadcast({ type: "stderr", text: d });
  });
  p.on("error", (e) => {
    engine.lastError = String(e.message || e);
    if (engine.proc === p) setState("error");
  });
  p.on("exit", (code, sig) => {
    if (engine.proc !== p) return;
    engine.proc = null;
    if (engine.run) finishRun({ cancelled: true, reason: sig ? `engine killed (${sig})` : `engine exited (${code})` });
    setState(code === 0 || sig ? "stopped" : "error", { exit: code, signal: sig });
  });
}

function stopEngine() {
  if (engine.proc) {
    const p = engine.proc;
    engine.proc = null;
    p.kill("SIGKILL");
  }
}

function onEngineOutput(p, chunk) {
  if (engine.proc !== p) return;
  const now = performance.now();
  if (engine.state === "starting") {
    engine.banner += chunk;
    const i = engine.banner.indexOf("\nYou: ");
    if (i >= 0) {
      engine.banner = engine.banner.slice(0, i).trim();
      engine.loadedMs = Date.now() - engine.startedAt;
      setState("ready", { loadedMs: engine.loadedMs });
    }
    return;
  }
  if (engine.state === "resetting") {
    engine.buf += chunk;
    if (engine.buf.includes("You: ")) {
      engine.buf = "";
      engine.history = [];
      setState("ready", { reset: true });
    }
    return;
  }
  if (engine.state !== "generating" || !engine.run) return;
  const r = engine.run;
  engine.buf += chunk;
  if (!r.started) {
    const a = engine.buf.indexOf("AI: ");
    if (a < 0) return;
    engine.buf = engine.buf.slice(a + 4);
    r.started = true;
  }
  const m = engine.buf.match(STATS_RE);
  if (m) {
    const text = engine.buf.slice(0, m.index);
    emitText(r, text, now);
    engine.buf = "";
    finishRun({
      stats: { promptTokens: +m[1], reused: +m[2], prefillTokens: +m[3], prefillS: +m[4], replyTokens: +m[5],
               decodeS: +m[6], decodeTps: +m[7], end: m[8] },
    });
    return;
  }
  // Errors from the CLI ("  error: ...") end the turn too.
  const err = engine.buf.match(/\n {2}error: (.*)\n/);
  if (err) {
    emitText(r, engine.buf.slice(0, err.index), now);
    engine.buf = "";
    finishRun({ error: err[1] });
    return;
  }
  // Stream everything except a tail that could be the start of the stats line.
  let hold = 0;
  for (let k = Math.min(STATS_HEAD.length, engine.buf.length); k > 0; --k) {
    if (STATS_HEAD.startsWith(engine.buf.slice(-k))) {
      hold = k;
      break;
    }
  }
  emitText(r, engine.buf.slice(0, engine.buf.length - hold), now);
  engine.buf = engine.buf.slice(engine.buf.length - hold);
}

function emitText(r, text, now) {
  if (!text) return;
  if (r.firstTokenAt === null) r.firstTokenAt = now;
  r.chunks += 1;
  r.text += text;
  r.lastAt = now;
  broadcast({ type: "token", runId: r.id, text, t: now - r.t0, chunks: r.chunks });
}

function finishRun(result) {
  const r = engine.run;
  if (!r) return;
  engine.run = null;
  const msg = { type: "done", runId: r.id, text: r.text, wallMs: performance.now() - r.t0,
                ttftMs: r.firstTokenAt === null ? null : r.firstTokenAt - r.t0, chunks: r.chunks, ...result };
  if (!result.cancelled) engine.history.push({ user: r.prompt, assistant: r.text });
  broadcast(msg);
  if (engine.proc && !result.cancelled) setState("ready");
}

let runSeq = 0;
function startRun(prompt) {
  if (engine.state !== "ready" || !engine.proc) return { error: `engine is ${engine.state}` };
  const line = prompt.replace(/\r?\n/g, " ").trim();
  if (!line) return { error: "empty prompt" };
  if (line.startsWith("/")) return { error: "prompts starting with / are engine commands" };
  engine.run = { id: ++runSeq, prompt: line, t0: performance.now(), firstTokenAt: null, lastAt: null, chunks: 0, text: "",
                 started: false };
  engine.buf = "";
  setState("generating", { runId: engine.run.id });
  broadcast({ type: "run", runId: engine.run.id, prompt: line, multiLineJoined: line !== prompt.trim() });
  engine.proc.stdin.write(line + "\n");
  return { ok: true };
}

// ---------------------------------------------------------------- /proc sampling (real OS data)
const prev = { threads: new Map(), cpus: null, t: 0 };

function readCpuTimes() {
  const out = [];
  for (const line of fs.readFileSync("/proc/stat", "utf8").split("\n")) {
    const m = line.match(/^cpu(\d+)\s+(.*)$/);
    if (!m) continue;
    const v = m[2].trim().split(/\s+/).map(Number);
    const idle = v[3] + (v[4] || 0);
    const total = v.reduce((a, b) => a + b, 0);
    out[+m[1]] = { idle, total };
  }
  return out;
}

function sample() {
  const t = performance.now();
  const dt = prev.t ? (t - prev.t) / 1000 : 0;
  const s = { type: "sample", t: Date.now(), state: engine.state, runId: engine.run ? engine.run.id : null,
              pid: engine.proc ? engine.proc.pid : null, threads: [],
              cpus: [], rssKiB: null, hwmKiB: null, memAvailKiB: null };
  // per logical CPU load (whole system)
  try {
    const cpus = readCpuTimes();
    if (prev.cpus) {
      s.cpus = cpus.map((c, i) => {
        const p = prev.cpus[i];
        if (!p || c.total === p.total) return 0;
        return Math.max(0, Math.min(100, (100 * (c.total - p.total - (c.idle - p.idle))) / (c.total - p.total)));
      });
    }
    prev.cpus = cpus;
  } catch {}
  try {
    const mi = fs.readFileSync("/proc/meminfo", "utf8");
    s.memAvailKiB = +(mi.match(/MemAvailable:\s+(\d+)/) || [])[1] || null;
    s.memTotalKiB = +(mi.match(/MemTotal:\s+(\d+)/) || [])[1] || null;
  } catch {}
  const p = engine.proc;
  if (p && p.pid) {
    try {
      const st = fs.readFileSync(`/proc/${p.pid}/status`, "utf8");
      s.rssKiB = +(st.match(/VmRSS:\s+(\d+)/) || [])[1] || null;
      s.hwmKiB = +(st.match(/VmHWM:\s+(\d+)/) || [])[1] || null;
      const seen = new Map();
      for (const tid of fs.readdirSync(`/proc/${p.pid}/task`)) {
        let raw;
        try {
          raw = fs.readFileSync(`/proc/${p.pid}/task/${tid}/stat`, "utf8");
        } catch {
          continue;
        }
        const rest = raw.slice(raw.lastIndexOf(")") + 2).split(" ");
        const ticks = Number(rest[11]) + Number(rest[12]); // utime + stime
        const cpu = Number(rest[36]); // processor last executed on
        const old = prev.threads.get(tid);
        const util = old && dt > 0 ? Math.max(0, Math.min(100, (100 * (ticks - old.ticks)) / (CLK_TCK * dt))) : 0;
        seen.set(tid, { ticks, cpu });
        s.threads.push({ tid: +tid, main: +tid === p.pid, state: rest[0], cpu, util, cpuSeconds: ticks / CLK_TCK,
                         migrated: old ? old.cpu !== cpu : false });
      }
      prev.threads = seen;
      s.threads.sort((a, b) => a.tid - b.tid);
    } catch {}
  } else {
    prev.threads = new Map();
  }
  prev.t = t;
  if (clients.size) broadcast(s);
}

// ---------------------------------------------------------------- operator profile (separate measured run)
let profiling = null;
function runProfile() {
  if (profiling) return { error: "a profile is already running" };
  if (engine.state === "generating") return { error: "wait for the reply to finish" };
  const s = engine.settings;
  const a = ["profile"];
  if (s.model === "bf16") a.push("--model-dir", CFG.modelDir);
  else a.push("--qmodel", CFG.efq);
  if (s.threads > 0) a.push("--threads", String(s.threads));
  if (s.kernels !== "auto") a.push("--kernels", s.kernels);
  broadcast({ type: "profile", state: "running", args: a });
  let out = "";
  profiling = spawn(CFG.bin, a, { cwd: ROOT });
  profiling.stdout.on("data", (d) => (out += d));
  profiling.stderr.on("data", (d) => (out += d));
  profiling.on("exit", (code) => {
    profiling = null;
    const ops = [];
    for (const line of out.split("\n")) {
      const m = line.match(/^ {2}([a-zA-Z/_ -]+?)\s{2,}([\d.]+)\s+([\d.]+)%(?:\s+([\d.]+ [KMG]iB)\s+([\d.]+))?\s*$/);
      if (m && !/^total/.test(m[1])) ops.push({ op: m[1].trim(), ms: +m[2], share: +m[3], weights: m[4] || null, gbps: m[5] ? +m[5] : null });
    }
    const total = out.match(/total \(forward\)\s+([\d.]+)\s+[\d.]+%\s+([\d.]+ [KMG]iB)\s+([\d.]+)/);
    const tps = out.match(/([\d.]+) tok\/s inside forward\(\)/);
    const head = out.match(/^\s+(\S+ kernels, \d+ threads? \| .*)$/m);
    broadcast({ type: "profile", state: code === 0 ? "done" : "error", ops, at: Date.now(), raw: out.slice(-4000),
                totalMs: total ? +total[1] : null, weightsPerToken: total ? total[2] : null, gbps: total ? +total[3] : null,
                tps: tps ? +tps[1] : null, setup: head ? head[1] : "" });
  });
  return { ok: true };
}

// ---------------------------------------------------------------- HTTP + WebSocket (built-ins only)
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
               ".svg": "image/svg+xml", ".json": "application/json" };

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (url.pathname === "/api/info") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(info()));
    return;
  }
  let file = path.normalize(path.join(DIST, url.pathname === "/" ? "index.html" : url.pathname));
  if (!file.startsWith(DIST) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) file = path.join(DIST, "index.html");
  if (!fs.existsSync(file)) {
    res.writeHead(500);
    res.end("studio/web/dist is missing: build the web app (studio/web: node build.mjs)");
    return;
  }
  res.writeHead(200, { "content-type": MIME[path.extname(file)] || "application/octet-stream", "cache-control": "no-cache" });
  fs.createReadStream(file).pipe(res);
});

function info() {
  return { type: "info", sys: SYS, logical: LOGICAL, clkTck: CLK_TCK,
           files: { modelDir: CFG.modelDir, safetensors: fileSize(path.join(CFG.modelDir, "model.safetensors")), efq: CFG.efq,
                    efqSize: fileSize(CFG.efq), bin: CFG.bin },
           historical: historical(), engine: { state: engine.state, settings: engine.settings, banner: engine.banner,
                                                args: engine.args, pid: engine.proc ? engine.proc.pid : null },
           instrumented: { schedulerJobs: false, perOpLive: false } };
}

server.on("upgrade", (req, socket) => {
  if (new URL(req.url, "http://localhost").pathname !== "/ws" || !req.headers["sec-websocket-key"]) {
    socket.destroy();
    return;
  }
  const accept = createHash("sha1").update(req.headers["sec-websocket-key"] + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").digest("base64");
  socket.write("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
               `Sec-WebSocket-Accept: ${accept}\r\n\r\n`);
  socket.setNoDelay(true);
  const client = {
    send(text) {
      const payload = Buffer.from(text);
      let header;
      if (payload.length < 126) header = Buffer.from([0x81, payload.length]);
      else if (payload.length < 65536) {
        header = Buffer.alloc(4);
        header[0] = 0x81;
        header[1] = 126;
        header.writeUInt16BE(payload.length, 2);
      } else {
        header = Buffer.alloc(10);
        header[0] = 0x81;
        header[1] = 127;
        header.writeBigUInt64BE(BigInt(payload.length), 2);
      }
      if (!socket.destroyed) socket.write(Buffer.concat([header, payload]));
    },
  };
  clients.add(client);
  let pending = Buffer.alloc(0);
  socket.on("data", (data) => {
    pending = Buffer.concat([pending, data]);
    while (pending.length >= 2) {
      const op = pending[0] & 0x0f;
      let len = pending[1] & 0x7f, off = 2;
      if (len === 126) {
        if (pending.length < 4) return;
        len = pending.readUInt16BE(2);
        off = 4;
      } else if (len === 127) {
        if (pending.length < 10) return;
        len = Number(pending.readBigUInt64BE(2));
        off = 10;
      }
      const masked = (pending[1] & 0x80) !== 0;
      const need = off + (masked ? 4 : 0) + len;
      if (pending.length < need) return;
      const mask = masked ? pending.subarray(off, off + 4) : null;
      const body = Buffer.from(pending.subarray(off + (masked ? 4 : 0), need));
      if (mask) for (let i = 0; i < body.length; ++i) body[i] ^= mask[i & 3];
      pending = pending.subarray(need);
      if (op === 0x8) {
        socket.end();
        return;
      }
      if (op === 0x9) continue; // ping: ignored (browsers do not send them)
      if (op === 0x1) onClientMessage(client, body.toString("utf8"));
    }
  });
  const drop = () => clients.delete(client);
  socket.on("close", drop);
  socket.on("error", drop);
  client.send(JSON.stringify(info()));
  client.send(JSON.stringify({ type: "engine", state: engine.state, settings: engine.settings, banner: engine.banner,
                               args: engine.args, pid: engine.proc ? engine.proc.pid : null, error: engine.lastError }));
});

function onClientMessage(client, text) {
  let m;
  try {
    m = JSON.parse(text);
  } catch {
    return;
  }
  const reply = (r) => client.send(JSON.stringify({ type: "ack", for: m.type, ...r }));
  if (m.type === "run") reply(startRun(String(m.prompt || "")));
  else if (m.type === "stop") {
    if (engine.run) {
      const r = engine.run;
      stopEngine();
      engine.run = null;
      broadcast({ type: "done", runId: r.id, text: r.text, cancelled: true, reason: "stopped by user",
                  wallMs: performance.now() - r.t0, ttftMs: r.firstTokenAt === null ? null : r.firstTokenAt - r.t0,
                  chunks: r.chunks });
      engine.history = [];
      startEngine(); // a fresh engine: the stopped conversation is gone
    }
    reply({ ok: true });
  } else if (m.type === "reset") {
    if (engine.state === "ready" && engine.proc) {
      engine.buf = "";
      setState("resetting");
      engine.proc.stdin.write("/reset\n");
      reply({ ok: true });
    } else reply({ error: `engine is ${engine.state}` });
  } else if (m.type === "configure") {
    engine.settings = { ...engine.settings, ...m.settings };
    engine.history = [];
    startEngine();
    reply({ ok: true });
  } else if (m.type === "start") {
    if (!engine.proc) startEngine();
    reply({ ok: true });
  } else if (m.type === "profile") reply(runProfile());
  else if (m.type === "info") client.send(JSON.stringify(info()));
}

// ---------------------------------------------------------------- start
const SYS = sysinfo();
for (const [k, v] of Object.entries({ bin: CFG.bin, efq: CFG.efq, "model dir": CFG.modelDir })) {
  if (!fs.existsSync(v)) console.error(`warning: ${k} not found: ${v}`);
}
server.listen(CFG.port, CFG.host, () => {
  console.log(`EdgeForge Studio: http://localhost:${CFG.port}  (bound to ${CFG.host}; engine ${CFG.bin})`);
  if (CFG.host !== "127.0.0.1" && CFG.host !== "localhost") console.log("warning: not bound to localhost only");
});
setInterval(sample, 100);
startEngine();
process.on("SIGINT", () => {
  stopEngine();
  process.exit(0);
});
process.on("SIGTERM", () => {
  stopEngine();
  process.exit(0);
});
