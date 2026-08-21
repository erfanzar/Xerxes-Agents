import { connect, type Socket } from "node:net";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
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
      while (i >= 0) {
        const line = this.buffer.slice(0, i);
        this.buffer = this.buffer.slice(i + 1);
        if (line.trim()) this.handle(JSON.parse(line) as Frame);
        i = this.buffer.indexOf("\n");
      }
    });
  }
  static async connect(p: string) {
    const s = connect({ path: p });
    await new Promise<void>((res, rej) => { s.once("connect", res); s.once("error", rej); });
    return new Client(s);
  }
  send(f: Record<string, unknown>) { this.socket.write(`${JSON.stringify(f)}\n`); }
  next(pred: (f: Frame) => boolean): Promise<Frame> {
    const idx = this.frames.findIndex(pred);
    if (idx >= 0) return Promise.resolve(this.frames.splice(idx, 1)[0]!);
    return new Promise((res) => this.waiters.push({ predicate: pred, resolve: res }));
  }
  handle(f: Frame) {
    const idx = this.waiters.findIndex(w => w.predicate(f));
    if (idx >= 0) { this.waiters.splice(idx, 1)[0]!.resolve(f); return; }
    this.frames.push(f);
  }
  close() { this.socket.destroy(); }
}

async function writeTranscript(path: string, sessionId: string, projectRoot: string, updatedAt: string, metadata: Record<string, unknown>) {
  await writeFile(join(path, `${sessionId}.json`), JSON.stringify({
    format: "xerxes-daemon-session", schema_version: 2, session_id: sessionId, key: sessionId,
    agent_id: "default", cwd: projectRoot, workspace: "", updated_at: updatedAt,
    messages: [{ role: "user", content: `request ${sessionId}` }, { role: "assistant", content: `response ${sessionId}` }],
    turn_count: 1, interaction_mode: "code", plan_mode: false,
    total_input_tokens: 1, total_output_tokens: 1,
    metadata: { project_root: projectRoot, ...metadata },
    thinking_content: [], tool_executions: [],
  }));
}

async function main() {
  const dir = await mkdtemp(join(tmpdir(), "xerxes-repro-"));
  const projectA = join(dir, "project-a");
  const sessionDir = join(dir, "sessions");
  const skillDir = join(dir, "skills");
  const socketPath = join(dir, "daemon.sock");
  await mkdir(sessionDir, { recursive: true });
  await mkdir(skillDir, { recursive: true });
  await writeTranscript(sessionDir, "aaaabbbb0001", projectA, "2026-07-17T00:02:00.000Z", { model: "root-model", title: "Project root" });
  await writeTranscript(sessionDir, "aaaabbbb0002", projectA, "2026-07-17T00:01:00.000Z", { parent_session_id: "aaaabbbb0001", title: "Regular branch" });
  await writeTranscript(sessionDir, "ccccdddd0001", projectA, "2026-07-17T00:03:00.000Z", {
    model: "child-model", parent_session_id: "aaaabbbb0001", root_session_id: "aaaabbbb0001",
    session_kind: "subagent", status: "completed", subagent_id: "subagent_child_one", title: "Child history",
  });

  const server = new DaemonServer({ socketPath, skillDirectory: skillDir, runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: projectA, sessionDirectory: sessionDir }) });
  await server.start();
  const client = await Client.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { project_dir: projectA, session_key: "session-list" } });
    await client.next(f => f.id === 1);
    await client.next(f => f.method === "event" && f.params?.type === "init_done");
    await client.next(f => f.method === "event" && f.params?.type === "status_update");

    client.send({ jsonrpc: "2.0", id: 7, method: "initialize", params: { project_dir: projectA, resume_session_id: "ccccdddd0001" } });
    const init7 = await client.next(f => f.id === 7);
    console.log("init7 error:", init7.error);
    console.log("init7 session:", init7.result?.session);
  } finally {
    client.close(); await server.stop(); await rm(dir, { recursive: true, force: true });
  }
}
main().catch(e => { console.error(e); process.exit(1); });
