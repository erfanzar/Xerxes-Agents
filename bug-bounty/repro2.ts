import { connect, type Socket } from "node:net";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DaemonServer } from "../xerxes/src/daemon/server.js";
import { InMemoryDaemonRuntime } from "../xerxes/src/daemon/runtime.js";

interface Frame {
  id?: number;
  method?: string;
  params?: { type?: string; payload?: Record<string, unknown> };
  result?: Record<string, unknown>;
  error?: { code?: number; message?: string };
}
class Client {
  private buffer = "";
  private frames: Frame[] = [];
  private waiters: Array<{ predicate: (f: Frame) => boolean; resolve: (f: Frame) => void }> = [];
  constructor(private socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", (chunk) => {
      this.buffer += (typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk));
      let i = this.buffer.indexOf("\n");
      while (i >= 0) { const line = this.buffer.slice(0, i); this.buffer = this.buffer.slice(i + 1); if (line.trim()) this.handle(JSON.parse(line) as Frame); i = this.buffer.indexOf("\n"); }
    });
  }
  static async connect(p: string) { const s = connect({ path: p }); await new Promise<void>((res, rej) => { s.once("connect", res); s.once("error", rej); }); return new Client(s); }
  send(f: Record<string, unknown>) { this.socket.write(`${JSON.stringify(f)}\n`); }
  next(pred: (f: Frame) => boolean): Promise<Frame> { const idx = this.frames.findIndex(pred); if (idx >= 0) return Promise.resolve(this.frames.splice(idx, 1)[0]!); return new Promise((res) => this.waiters.push({ predicate: pred, resolve: res })); }
  handle(f: Frame) { const idx = this.waiters.findIndex(w => w.predicate(f)); if (idx >= 0) { this.waiters.splice(idx, 1)[0]!.resolve(f); return; } this.frames.push(f); }
  close() { this.socket.destroy(); }
}

async function main() {
  const dir = await mkdtemp(join(tmpdir(), "xerxes-repro2-"));
  const skillDir = join(dir, "skills");
  const socketPath = join(dir, "daemon.sock");
  await mkdir(skillDir, { recursive: true });
  const server = new DaemonServer({ socketPath, skillDirectory: skillDir, runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: dir, model: "protocol-model", sessionDirectory: join(dir, "sessions") }) });
  await server.start();
  const client = await Client.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "runtime.status", params: {} });
    console.log("status:", (await client.next(f => f.id === 1)).result?.model, (await client.next(f => f.id === 1)).result?.provider);
    client.send({ jsonrpc: "2.0", id: 2, method: "initialize", params: { session_key: "test-session" } });
    const init2 = await client.next(f => f.id === 2);
    console.log("init2 error:", init2.error);
  } finally { client.close(); await server.stop(); await rm(dir, { recursive: true, force: true }); }
}
main().catch(e => { console.error(e); process.exit(1); });
