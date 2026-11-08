# EdgeForge Studio (sprint MVP)

A browser UI on top of the real EdgeForge engine: type a prompt, the C++ engine streams TinyLlama's
answer, and you see the engine's threads working live.

```bash
# WSL, once: Node.js 18+ (no npm install needed: the web app in studio/web/dist is prebuilt)
sudo apt install -y nodejs
# start (defaults: ~/edgeforge-data/tinyllama, ~/edgeforge-data/tinyllama-q4.efq, ~/ef-build/edgeforge)
cd "/mnt/c/Users/BIT/OneDrive/Desktop/bios Hackathon"
bash studio/run.sh
# then open http://localhost:8790 in the Windows browser
```

If the Windows browser cannot reach it, run `HOST=0.0.0.0 bash studio/run.sh`. This listens on
WSL's network too, not only on localhost.

## What you see

- **Left: prompt and answer.** Run, Stop, Clear and Reset. The answer streams in live.
  - The engine runs with fixed defaults: the Q4 `.efq` model, automatic kernels and thread count,
    greedy decoding, at most 256 new tokens, and the engine's default system prompt.
- **Right, top (about two thirds): live CPU and thread utilization.**
  - One tall bar per logical vCPU: system-wide load from `/proc/stat`, every 100 ms.
  - Below, every real EdgeForge engine thread: its CPU % and the Linux vCPU it last ran on, from
    `/proc/<pid>/task`. By default the engine starts one thread per CPU core, so the Ryzen 5 7530U
    (6 cores, 12 vCPUs) shows 6 thread rows. On this laptop 6 threads decoded faster than 12 (21.9
    vs 19.4 tok/s, Q4 avx2, `bench/report_DESKTOP-IH5KR28.md`).
  - Every thread always gets a visible row. More than six threads, or a short window, switch the
    list to two columns; the vCPU bars shrink before any thread row is hidden.
- **Right, bottom: five numbers.** Tokens, reused tokens, computed tokens, processing time and
  tokens/sec.
  - While a reply streams, tokens, time and tokens/sec are live; tokens and tokens/sec are marked
    ≈ (estimates).
  - When the reply ends, all five are the engine's own final values.

## How it works

- **Bridge (`studio/server.mjs`, Node built-ins only).**
  - It runs the unchanged `edgeforge chat` CLI as a child process and talks to it over stdin/stdout.
  - **Stop** kills the engine and starts a fresh one, so that conversation is cleared.
- **Web app (`studio/web`).** React + TypeScript, bundled with esbuild (`npm install && node
  build.mjs` rebuilds `dist/`). The editor is Monaco from the jsDelivr CDN; without internet it
  falls back to a plain editor.
- **Look.** Light steel panels; CPU load is drawn as heated metal (dark red when lightly loaded,
  orange to amber near 100%). Fonts are Big Shoulders Display and Archivo from Google Fonts; offline,
  Windows falls back to its built-in Bahnschrift and Segoe UI.

## Honest limits

- **Per-job scheduler events are not shown.** These would be the tensor, the row range and the
  layer of each thread-pool job; the engine does not export them yet. The thread rows show real OS
  activity: Worker 0 is the main thread (tokenizer, sampling, and chunk 0 of every parallel job), and
  Workers 1… are the pool threads.
- **In WSL2, "vCPU N" is the Linux virtual CPU.** It is not a Windows physical core.
- **Live tokens and tokens/sec are estimates.** They count stream events (about one per token). Reused
  and computed tokens are only known when the reply ends.
- **Prompts are sent as one line.** The CLI reads one line per turn, so multi-line prompts are joined.
