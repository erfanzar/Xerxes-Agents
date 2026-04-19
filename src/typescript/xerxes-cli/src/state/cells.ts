import type { Reducer } from "react";

export type Cell =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string; streaming: boolean }
  | { kind: "thinking"; id: string; text: string; streaming: boolean }
  | ToolCell
  | { kind: "system"; id: string; text: string }
  | { kind: "error"; id: string; text: string }
  | SubAgentCell;

export interface ToolCell {
  kind: "tool";
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  durationMs?: number;
  permitted?: boolean;
  tool_call_id?: string;
}

export interface SubAgentCell {
  kind: "subagent";
  id: string;
  name: string;
  status: "running" | "completed" | "failed" | "cancelled";
  text: string;
  streaming: boolean;
}

export type CellsAction =
  | { type: "push"; cell: Cell }
  | { type: "append_assistant"; text: string }
  | { type: "append_thinking"; text: string }
  | { type: "finalize_assistant" }
    | {
        type: "tool_end";
        name: string;
        result: string;
        durationMs: number;
        permitted: boolean;
        tool_call_id?: string;
      }
  | { type: "clear" }
  | { type: "append_subagent_text"; name: string; text: string }
  | { type: "finalize_subagent"; name: string; status: "completed" | "failed" | "cancelled" };

// Use a closure with a mutable ref to avoid global state issues
// while still providing unique IDs per session
const newIdFactory = () => {
  let nextId = 0;
  return {
    next: () => `c${++nextId}`,
    reset: () => { nextId = 0; }
  };
};
const idGen = newIdFactory();
const newId = () => idGen.next();

export const cellsReducer: Reducer<Cell[], CellsAction> = (state, action) => {
  switch (action.type) {
    case "push":
      return [...state, { ...action.cell, id: action.cell.id || newId() }];
    case "append_assistant": {
      // Find the most recent streaming assistant cell and append to it,
      // even if tool or thinking cells were pushed in between.
      for (let i = state.length - 1; i >= 0; i--) {
        const c = state[i];
        if (c.kind === "assistant" && c.streaming) {
          const updated: Cell = { ...c, text: c.text + action.text };
          return state.map((x, j) => (j === i ? updated : x));
        }
      }
      return [
        ...state,
        { kind: "assistant", id: newId(), text: action.text, streaming: true },
      ];
    }
    case "append_thinking": {
      for (let i = state.length - 1; i >= 0; i--) {
        const c = state[i];
        if (c.kind === "thinking" && c.streaming) {
          const updated: Cell = { ...c, text: c.text + action.text };
          return state.map((x, j) => (j === i ? updated : x));
        }
      }
      return [
        ...state,
        { kind: "thinking", id: newId(), text: action.text, streaming: true },
      ];
    }
    case "finalize_assistant": {
      // Finalize ALL streaming assistant and thinking cells, not just the last one.
      return state.map((c) =>
        (c.kind === "assistant" || c.kind === "thinking") && c.streaming
          ? ({ ...c, streaming: false } as Cell)
          : c,
      );
    }
    case "tool_end": {
      const { name, result, durationMs, permitted, tool_call_id } = action;
      // prefer tool_call_id, fall back to name for backward compat
      for (let i = state.length - 1; i >= 0; i--) {
        const c = state[i];
        if (c.kind !== "tool" || c.result !== undefined) continue;
        
        // Check match based on tool_call_id or name
        const matches = tool_call_id
          ? (c as ToolCell).tool_call_id === tool_call_id
          : c.name === name;
        
        if (matches) {
          const updated: ToolCell = {
            ...c,
            result,
            durationMs,
            permitted,
          };
          return state.map((x, j) => (j === i ? updated : x));
        }
      }
      return state;
    }
    case "append_subagent_text": {
      const { name, text } = action;
      const MAX_SUBAGENT_CHARS = 100;
      for (let i = state.length - 1; i >= 0; i--) {
        const c = state[i];
        if (c.kind === "subagent" && c.name === name && c.streaming) {
          let newText = c.text + text;
          if (newText.length > MAX_SUBAGENT_CHARS) {
            newText = "… " + newText.slice(-MAX_SUBAGENT_CHARS);
          }
          const updated: SubAgentCell = { ...c, text: newText };
          return state.map((x, j) => (j === i ? updated : x));
        }
      }
      // No matching running sub-agent; push a new one
      const initialText =
        text.length > MAX_SUBAGENT_CHARS ? "… " + text.slice(-MAX_SUBAGENT_CHARS) : text;
      return [
        ...state,
        { kind: "subagent", id: newId(), name, status: "running", text: initialText, streaming: true },
      ];
    }
    case "finalize_subagent": {
      const { name, status } = action;
      for (let i = state.length - 1; i >= 0; i--) {
        const c = state[i];
        if (c.kind === "subagent" && c.name === name && c.streaming) {
          const updated: SubAgentCell = { ...c, streaming: false, status };
          return state.map((x, j) => (j === i ? updated : x));
        }
      }
      return state;
    }
    case "clear":
      idGen.reset();
      return [];
    default:
      return state;
  }
};

export const mkCell = (cell: Omit<Cell, "id"> & { id?: string }): Cell => ({
  ...cell,
  id: cell.id || newId(),
}) as Cell;
