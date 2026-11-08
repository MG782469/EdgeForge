// EdgeForge Studio - web UI. Every number on screen is either measured by the OS (/proc, sampled by
// the bridge every 100 ms) or reported by the EdgeForge engine itself; live estimates are marked.
import { createRoot } from "react-dom/client";
import { Marked } from "marked";
import React, { useEffect, useRef, useState } from "react";

// ------------------------------------------------------------------ types
type ThreadSample = { tid: number; main: boolean; state: string; cpu: number; util: number; cpuSeconds: number; migrated: boolean };
type Sample = { type: "sample"; t: number; state: string; pid?: number | null; threads: ThreadSample[]; cpus: number[] };
type Stats = {
  promptTokens: number;
  reused: number;
  prefillTokens: number;
  prefillS: number;
  replyTokens: number;
  decodeS: number;
  decodeTps: number;
  end: string;
};
type Msg = { role: "user" | "assistant" | "note"; text: string; runId?: number; stats?: Stats; cancelled?: boolean; error?: string; streaming?: boolean };
type Live = { runId: number; t0: number; chunks: number; firstAt: number | null; lastAt: number };
type Info = { sys: Record<string, any>; logical: number };

function save(key: string, v: unknown) {
  try {
    localStorage.setItem(key, typeof v === "string" ? v : JSON.stringify(v));
  } catch {}
}
function loadText(key: string, def: string) {
  try {
    return localStorage.getItem(key) ?? def;
  } catch {
    return def;
  }
}

const md = new Marked({ gfm: true, breaks: true });
md.use({ renderer: { html: (t: any) => escapeHtml(typeof t === "string" ? t : t.text) } });
function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

// ------------------------------------------------------------------ bridge connection
function useBridge() {
  const [connected, setConnected] = useState(false);
  const [info, setInfo] = useState<Info | null>(null);
  const [engine, setEngine] = useState<any>({ state: "connecting" });
  const [sample, setSample] = useState<Sample | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [live, setLive] = useState<Live | null>(null);
  const [stderr, setStderr] = useState("");
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let stop = false;
    let timer: any;
    const connect = () => {
      const sock = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.current = sock;
      sock.onopen = () => setConnected(true);
      sock.onclose = () => {
        setConnected(false);
        if (!stop) timer = setTimeout(connect, 1000);
      };
      sock.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        switch (m.type) {
          case "info":
            setInfo(m);
            break;
          case "engine":
            setEngine(m);
            if (m.reset) setMessages((x) => [...x, { role: "note", text: "Conversation cleared." }]);
            break;
          case "sample":
            setSample(m);
            break;
          case "run":
            setLive({ runId: m.runId, t0: performance.now(), chunks: 0, firstAt: null, lastAt: performance.now() });
            setMessages((x) => [
              ...x,
              ...(m.multiLineJoined ? [{ role: "note" as const, text: "Multi-line prompt sent as one line (the engine reads one line per turn)." }] : []),
              { role: "user", text: m.prompt, runId: m.runId },
              { role: "assistant", text: "", runId: m.runId, streaming: true },
            ]);
            break;
          case "token":
            setLive((l) => (l && l.runId === m.runId ? { ...l, chunks: m.chunks, firstAt: l.firstAt ?? performance.now(), lastAt: performance.now() } : l));
            setMessages((x) => x.map((msg) => (msg.role === "assistant" && msg.runId === m.runId ? { ...msg, text: msg.text + m.text } : msg)));
            break;
          case "done":
            setMessages((x) =>
              x.map((msg) =>
                msg.role === "assistant" && msg.runId === m.runId
                  ? { ...msg, text: m.text, streaming: false, stats: m.stats, cancelled: m.cancelled, error: m.error ?? (m.cancelled ? m.reason : undefined) }
                  : msg,
              ),
            );
            setLive(null);
            if (m.cancelled) setMessages((x) => [...x, { role: "note", text: "Stopped: the engine was restarted, so the conversation starts fresh." }]);
            break;
          case "stderr":
            setStderr((s) => (s + m.text).slice(-2000));
            break;
          case "ack":
            if (m.error) setMessages((x) => [...x, { role: "note", text: `Not sent: ${m.error}` }]);
            break;
        }
      };
    };
    connect();
    return () => {
      stop = true;
      clearTimeout(timer);
      ws.current?.close();
    };
  }, []);
  const send = (msg: unknown) => ws.current?.readyState === 1 && ws.current.send(JSON.stringify(msg));
  return { connected, info, engine, sample, messages, live, stderr, send };
}

// ------------------------------------------------------------------ editor (Monaco from CDN, plain editor offline)
function PromptEditor({ value, onChange, onRun }: { value: string; onChange: (v: string) => void; onRun: () => void }) {
  const host = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"loading" | "monaco" | "plain">("loading");
  const editor = useRef<any>(null);
  const runRef = useRef(onRun);
  runRef.current = onRun;
  useEffect(() => {
    let done = false;
    const fallback = setTimeout(() => !done && setMode("plain"), 6000);
    const base = "https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min/vs";
    const s = document.createElement("script");
    s.src = `${base}/loader.js`;
    s.onerror = () => setMode("plain");
    s.onload = () => {
      const w = window as any;
      w.require.config({ paths: { vs: base } });
      w.require(["vs/editor/editor.main"], () => {
        if (done || !host.current) return;
        done = true;
        clearTimeout(fallback);
        const ed = w.monaco.editor.create(host.current, {
          value,
          language: "markdown",
          theme: "vs",
          minimap: { enabled: false },
          wordWrap: "on",
          fontSize: 14,
          lineNumbers: "off",
          scrollBeyondLastLine: false,
          automaticLayout: true,
          padding: { top: 10 },
          renderLineHighlight: "none",
        });
        ed.onDidChangeModelContent(() => onChange(ed.getValue()));
        ed.addCommand(w.monaco.KeyMod.CtrlCmd | w.monaco.KeyCode.Enter, () => runRef.current());
        editor.current = ed;
        setMode("monaco");
      });
    };
    document.head.appendChild(s);
    return () => {
      done = true;
      clearTimeout(fallback);
      editor.current?.dispose();
    };
  }, []);
  useEffect(() => {
    if (editor.current && editor.current.getValue() !== value) editor.current.setValue(value);
  }, [value]);
  return (
    <div className="editor">
      <div className="editor-host" ref={host} style={{ display: mode === "monaco" ? "block" : "none" }} />
      {mode !== "monaco" && (
        <textarea
          className="plain-editor"
          value={value}
          spellCheck={false}
          placeholder="Ask TinyLlama something..."
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              onRun();
            }
          }}
        />
      )}
      <div className="editor-mode">Ctrl+Enter to run</div>
    </div>
  );
}

// ------------------------------------------------------------------ right upper: live CPU / thread utilization
// Heat fill: the gradient always spans the whole slot, so a busier bar shows more of the ramp.
function Heat({ pct, dir }: { pct: number; dir: "up" | "right" }) {
  const p = Math.max(0, Math.min(100, pct));
  const inner = p > 0 ? `${(100 * 100) / p}%` : "100%";
  return (
    <span className={`heat heat-${dir}`} style={dir === "up" ? { height: `${p}%` } : { width: `${p}%` }}>
      <span className="heat-ramp" style={dir === "up" ? { height: inner } : { width: inner }} />
    </span>
  );
}

function CpuPanel({ sample, info, running, autoThreads }: { sample: Sample | null; info: Info | null; running: boolean; autoThreads: boolean }) {
  const logical = info?.logical || sample?.cpus.length || 0;
  const wsl = !!info?.sys?.wsl;
  const cpuName = wsl ? "vCPU" : "CPU";
  const threads = sample?.pid ? sample.threads : [];
  const total = threads.reduce((a, t) => a + t.util, 0);
  return (
    <section className="plate cpu-plate">
      <header className="plate-head">
        <h2>CPU activity</h2>
        <span className={`state ${running ? "on" : ""}`}>{running ? "Generating" : "Idle"}</span>
        <span className="spacer" />
        <span className="quiet">{info?.sys?.cpu}</span>
      </header>
      <p className="sub">{wsl ? "Every logical vCPU Linux sees inside WSL2, all processes" : "Every logical CPU, all processes"}</p>
      <div className="vcpus" style={{ gridTemplateColumns: `repeat(${Math.max(logical, 1)}, minmax(0, 1fr))` }}>
        {Array.from({ length: logical }, (_, i) => {
          const u = sample?.cpus[i] ?? 0;
          return (
            <div className="vcpu" key={i} title={`${wsl ? "Observed Linux vCPU" : "CPU"} ${i}: ${u.toFixed(0)}% busy`}>
              <span className="vcpu-pct">{sample?.cpus.length ? u.toFixed(0) : "–"}<small>%</small></span>
              <div className="slot slot-up">
                <Heat pct={u} dir="up" />
              </div>
              <span className="vcpu-id"><span className="pre">{cpuName} </span>{i}</span>
            </div>
          );
        })}
      </div>
      <div className="threads-head">
        <h3>EdgeForge threads</h3>
        <span className="quiet">
          {threads.length
            ? `${threads.length} threads${autoThreads ? " (engine default: one per CPU core)" : ""}, ${(total / 100).toFixed(1)} cores' worth busy`
            : "engine not running"}
        </span>
      </div>
      {/* every engine thread gets a row; more than six go into two columns so none is ever hidden */}
      <div className={`workers ${threads.length > 6 ? "many" : ""}`} style={{ "--rows": Math.ceil(threads.length / 2) } as React.CSSProperties}>
        {threads.map((t, i) => (
          <div className="worker" key={t.tid} title={`${t.main ? "Worker 0 (main thread)" : `Worker ${i}`}, tid ${t.tid}: ${t.util.toFixed(0)}% busy, last ran on ${wsl ? "Linux vCPU" : "CPU"} ${t.cpu}`}>
            <span className="worker-name">
              {t.main ? "Worker 0, main" : `Worker ${i}`}
              <span className="quiet"> tid {t.tid}</span>
            </span>
            <span className="slot slot-right">
              <Heat pct={t.util} dir="right" />
            </span>
            <span className="worker-pct">{t.util.toFixed(0)}%</span>
            <span className="worker-cpu">
              <span className="pre">{wsl ? "on Linux " : "on "}</span>
              {wsl ? "vCPU" : "CPU"} {t.cpu}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ right lower: five metrics
function Metrics({ messages, live, now }: { messages: Msg[]; live: Live | null; now: number }) {
  const last = [...messages].reverse().find((m) => m.role === "assistant" && !m.streaming && m.stats);
  const st = last?.stats;
  const rate = live && live.firstAt && live.chunks > 1 ? (live.chunks - 1) / ((live.lastAt - live.firstAt) / 1000 || 1) : null;
  const items = live
    ? [
        { label: "Tokens", value: live.chunks ? `≈${live.chunks}` : "0", note: "live estimate" },
        { label: "Reused", value: "…", note: "known at the end" },
        { label: "Computed", value: "…", note: "known at the end" },
        { label: "Processing time", value: `${((now - live.t0) / 1000).toFixed(1)}s`, note: "elapsed so far" },
        { label: "Tokens/sec", value: rate ? `≈${rate.toFixed(1)}` : "…", note: "live estimate" },
      ]
    : st
      ? [
          { label: "Tokens", value: `${st.replyTokens}`, note: "generated" },
          { label: "Reused", value: `${st.reused}`, note: "from KV cache" },
          { label: "Computed", value: `${st.prefillTokens}`, note: "new prompt tokens" },
          { label: "Processing time", value: `${(st.prefillS + st.decodeS).toFixed(2)}s`, note: "prompt + decoding" },
          { label: "Tokens/sec", value: st.decodeTps.toFixed(2), note: "decoding speed" },
        ]
      : ["Tokens", "Reused", "Computed", "Processing time", "Tokens/sec"].map((label) => ({ label, value: "–", note: "" }));
  return (
    <section className="plate metrics-plate">
      <header className="plate-head">
        <h2>Overview</h2>
        <span className="quiet">{live ? "Live while the answer streams; the engine reports final values at the end" : st ? "Final values from the engine for the last answer" : "Run a prompt to see its numbers"}</span>
      </header>
      <div className="gauges">
        {items.map((it) => (
          <div className="gauge" key={it.label}>
            <span className="gauge-label">{it.label}</span>
            <span className="gauge-value">{it.value}</span>
            <span className="gauge-note">{it.note}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ app
function App() {
  const b = useBridge();
  const [prompt, setPrompt] = useState(() => loadText("ef.prompt", "Explain how a CPU cache works in two sentences."));
  const [now, setNow] = useState(performance.now());
  const out = useRef<HTMLDivElement>(null);
  useEffect(() => save("ef.prompt", prompt), [prompt]);
  useEffect(() => {
    out.current?.scrollTo({ top: out.current.scrollHeight });
  }, [b.messages]);
  useEffect(() => {
    if (!b.live) return;
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [b.live]);
  const st = b.engine.state as string;
  const busy = st === "generating";
  const run = () => !busy && st === "ready" && b.send({ type: "run", prompt });
  const runLabel = !b.connected ? "Studio offline" : st === "starting" || st === "connecting" ? "Loading model…" : st === "error" ? "Engine error" : "Run";
  return (
    <div className="app">
      <header className="top">
        <span className="mark" aria-hidden="true" />
        <span className="wordmark">EdgeForge Studio</span>
      </header>
      <main className="layout">
        <section className="plate left">
          <header className="plate-head">
            <h2>Ask TinyLlama</h2>
            <span className="spacer" />
            <button className="primary" onClick={run} disabled={busy || st !== "ready"}>{runLabel}</button>
            <button onClick={() => b.send({ type: "stop" })} disabled={!busy}>Stop</button>
            <button onClick={() => setPrompt("")}>Clear</button>
            <button onClick={() => b.send({ type: "reset" })} disabled={st !== "ready"} title="Start a new conversation">Reset</button>
          </header>
          <PromptEditor value={prompt} onChange={setPrompt} onRun={run} />
          <div className="transcript" ref={out}>
            {b.messages.length === 0 && <p className="empty">Write a prompt and press Run. The answer is generated on this laptop's CPU by the EdgeForge engine.</p>}
            {b.messages.map((m, i) =>
              m.role === "note" ? (
                <div key={i} className="note">{m.text}</div>
              ) : m.role === "user" ? (
                <div key={i} className="msg user">{m.text}</div>
              ) : (
                <div key={i} className="msg ai">
                  <div className="md" dangerouslySetInnerHTML={{ __html: md.parse(m.text || (m.streaming ? "…" : "")) as string }} />
                  {m.stats && <div className="stats">{m.stats.replyTokens} tokens at {m.stats.decodeTps.toFixed(1)} tokens per second, ended by {m.stats.end}</div>}
                  {m.error && <div className="stats warn">{m.error}</div>}
                </div>
              ),
            )}
          </div>
          {b.stderr && <pre className="raw">{b.stderr}</pre>}
        </section>
        <div className="right">
          <CpuPanel sample={b.sample} info={b.info} running={busy} autoThreads={b.engine.settings?.threads === 0} />
          <Metrics messages={b.messages} live={b.live} now={now} />
        </div>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
