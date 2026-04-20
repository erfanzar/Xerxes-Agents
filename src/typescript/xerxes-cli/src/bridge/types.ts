// JSON-RPC protocol shared with `python -m xerxes.bridge`.
// Request types are sent stdin → Python; Event types arrive stdout ← Python.

export type Request =
  | {
      method: "init";
      params: {
        model: string;
        permission_mode: string;
        base_url?: string;
        api_key?: string;
      };
    }
  | { method: "query"; params: { text: string } }
  | { method: "permission_response"; params: { granted: boolean } }
  | { method: "question_response"; params: { answer: string } }
  | { method: "slash"; params: { command: string } }
  | { method: "cancel"; params?: Record<string, never> }
  | { method: "shutdown"; params?: Record<string, never> }
  | { method: "provider_list"; params?: Record<string, never> }
  | { method: "fetch_models"; params: { base_url: string; api_key: string } }
  | {
      method: "provider_save";
      params: { name: string; base_url: string; api_key: string; model: string; provider?: string };
    }
  | { method: "provider_select"; params: { name: string } };

export type Event =
  | {
      event: "ready";
      data: {
        model: string;
        provider: string;
        tools: number;
        permission_mode: string;
        has_profile: boolean;
        skills: string[];
      };
    }
  | { event: "text_chunk"; data: { text: string } }
  | { event: "thinking_chunk"; data: { text: string } }
  | {
      event: "tool_start";
      data: {
        name: string;
        inputs: Record<string, unknown>;
        tool_call_id?: string;
      };
    }
  | {
      event: "tool_end";
      data: {
        name: string;
        result: string;
        permitted: boolean;
        duration_ms: number;
        tool_call_id?: string;
      };
    }
  | {
      event: "permission_request";
      data: { tool_name: string; description: string; arguments?: unknown };
    }
  | {
      event: "question_request";
      data: { question: string };
    }
  | {
      event: "turn_done";
      data: { input_tokens: number; output_tokens: number };
    }
  | { event: "slash_result"; data: { output: string } }
  | { event: "error"; data: { message: string } }
  | {
      event: "state";
      data: {
        turn_count: number;
        total_input_tokens: number;
        total_output_tokens: number;
        context_limit?: number;
        remaining_context?: number;
        cost_usd?: number;
      };
    }
  | {
      event: "provider_list";
      data: { profiles: ProviderProfile[] };
    }
  | {
      event: "models_list";
      data: { models: string[]; base_url: string };
    }
  | {
      event: "provider_saved";
      data: {
        profile: ProviderProfile;
        message: string;
      };
    }
  | { event: "model_changed"; data: { model: string; provider?: string } }
  | { event: "skills_updated"; data: { skills: string[] } }
  | { event: "query_done"; data: Record<string, never> }
  | { event: "exit"; data: Record<string, never> }
  | {
      event: "agent_spawn";
      data: {
        task_id: string;
        agent_name: string;
        agent_type: string;
        prompt?: string;
        depth?: number;
        isolation?: string;
      };
    }
  | {
      event: "agent_text";
      data: {
        task_id: string;
        agent_name: string;
        agent_type: string;
        text: string;
      };
    }
  | {
      event: "agent_thinking";
      data: {
        task_id: string;
        agent_name: string;
        agent_type: string;
        text: string;
      };
    }
  | {
      event: "agent_tool_start";
      data: {
        task_id: string;
        agent_name: string;
        agent_type: string;
        tool_name: string;
        inputs?: Record<string, unknown>;
      };
    }
  | {
      event: "agent_tool_end";
      data: {
        task_id: string;
        agent_name: string;
        agent_type: string;
        tool_name: string;
        result?: string;
        permitted?: boolean;
        duration_ms?: number;
      };
    }
  | {
      event: "agent_done";
      data: {
        task_id: string;
        agent_name: string;
        agent_type: string;
        status: string;
        result?: string;
      };
    };

export interface ProviderProfile {
  name: string;
  base_url: string;
  api_key?: string;
  model: string;
  provider?: string;
  active?: boolean;
  [key: string]: unknown;
}
