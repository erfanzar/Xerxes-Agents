// Spawn the Python bridge subprocess, expose a send() and an event stream.

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { EventEmitter } from "node:events";
import type { Event, Request } from "./types.js";

/** Python executable names tried in order when `--python` is unspecified. */
const PYTHON_FALLBACKS = [
  "python",
  "python3",
  "python3.13",
  "python3.12",
  "python3.11",
  "python3.10",
];

function existsOnPath(name: string): boolean {
  // If the user gave an absolute or relative path, just check the FS.
  if (name.includes("/") || (process.platform === "win32" && name.includes("\\"))) {
    try {
      return require("node:fs").existsSync(name);
    } catch {
      return false;
    }
  }
  const pathVar = process.env.PATH ?? "";
  const sep = process.platform === "win32" ? ";" : ":";
  const exts = process.platform === "win32" ? [".exe", ".bat", ".cmd", ""] : [""];
  const fs = require("node:fs");
  const pathMod = require("node:path");
  for (const dir of pathVar.split(sep)) {
    if (!dir) continue;
    for (const ext of exts) {
      const candidate = pathMod.join(dir, name + ext);
      try {
        const stat = fs.statSync(candidate);
        if (stat.isFile()) return true;
      } catch {
        // not found
      }
    }
  }
  return false;
}

function hasXerxes(pythonCmd: string): boolean {
  try {
    const { execSync } = require("node:child_process");
    execSync(`${pythonCmd} -c "import xerxes"`, {
      stdio: ["ignore", "ignore", "ignore"],
      timeout: 5000,
    });
    return true;
  } catch {
    return false;
  }
}

export function resolvePython(requested: string): string {
  if (existsOnPath(requested)) {
    if (requested.includes("/") || (process.platform === "win32" && requested.includes("\\"))) {
      if (hasXerxes(requested)) return requested;
    } else if (hasXerxes(requested)) return requested;
  }
  for (const cand of PYTHON_FALLBACKS) {
    if (cand !== requested && existsOnPath(cand) && hasXerxes(cand)) return cand;
  }
  const home = require("node:os").homedir();
  const venvCandidates = [
    `${home}/Documents/Projects/Xerxes-Agents/.venv/bin/python3`,
    `${home}/.local/share/uv/tools/xerxes/bin/python`,
    `${home}/.venv/bin/python`,
  ];
  for (const venv of venvCandidates) {
    try {
      if (require("node:fs").existsSync(venv) && hasXerxes(venv)) return venv;
    } catch {
      // continue
    }
  }
  return requested;
}

export interface Bridge {
  send(req: Request): void;
  on(listener: (event: Event) => void): () => void;
  close(): void;
}

export function spawnBridge(opts: {
  python: string;
  projectDir: string;
}): Bridge {
  const pythonCmd = resolvePython(opts.python);
  const child = spawn(pythonCmd, ["-m", "xerxes.bridge"], {
    cwd: opts.projectDir,
    stdio: ["pipe", "pipe", "ignore"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  const stdin = child.stdin!;
  const stdout = child.stdout!;

  const emitter = new EventEmitter();

  const rl = createInterface({ input: stdout });
  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    
    // Try to parse the whole line first
    try {
      const parsed = JSON.parse(trimmed) as Event;
      emitter.emit("event", parsed);
      return;
    } catch (lineErr) {
      // Line-level parse failed; try splitting by whitespace to handle
      // multiple JSON objects on one line (common with rapid streaming).
      const tokens = trimmed.split(/\s+/);
      for (const token of tokens) {
        if (!token.trim()) continue;
        try {
          const parsed = JSON.parse(token) as Event;
          emitter.emit("event", parsed);
        } catch (tokenErr) {
          emitter.emit("event", {
            event: "error",
            data: { message: `Bridge parse error: ${String(tokenErr)} — ${token.slice(0, 200)}` },
          } as Event);
        }
      }
    }
  });

  child.on("error", (err) => {
    emitter.emit("event", {
      event: "error",
      data: {
        message:
          `Failed to spawn Python bridge with '${pythonCmd}': ${err.message}. ` +
          `Try --python python3 or activate a venv.`,
      },
    } as Event);
  });

  child.on("exit", (code, signal) => {
    if (code !== 0 && signal !== "SIGTERM") {
      emitter.emit("event", {
        event: "error",
        data: { message: `Python bridge exited: code=${code} signal=${signal}` },
      } as Event);
    }
  });

  return {
    send(req) {
      try {
        stdin.write(JSON.stringify(req) + "\n");
      } catch (err) {
        emitter.emit("event", {
          event: "error",
          data: { message: `Bridge send failed: ${String(err)}` },
        } as Event);
      }
    },
    on(listener) {
      emitter.on("event", listener);
      return () => emitter.off("event", listener);
    },
    close() {
      try {
        stdin.end();
      } catch {
        // ignore
      }
      setTimeout(() => {
        try {
          child.kill("SIGTERM");
        } catch {
          // ignore
        }
      }, 200);
    },
  };
}
