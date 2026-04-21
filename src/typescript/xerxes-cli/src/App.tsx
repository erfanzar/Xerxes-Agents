import React, { useEffect, useReducer, useState, useRef } from "react";
import { Box, useApp, useInput } from "ink";
import type { CliArgs } from "./utils/args.js";
import { Header } from "./components/Header.js";
import { Cells } from "./components/Cells.js";
import { Composer } from "./components/Composer.js";
import { Footer } from "./components/Footer.js";
import { PermissionPrompt } from "./components/PermissionPrompt.js";
import { QuestionPrompt } from "./components/QuestionPrompt.js";
import {
  ProviderDialog,
  type ProviderStep,
} from "./components/ProviderDialog.js";
import { useBridge } from "./hooks/useBridge.js";
import { useSpinner } from "./hooks/useSpinner.js";
import { cellsReducer } from "./state/cells.js";
import type { Event, ProviderProfile } from "./bridge/types.js";

interface AppProps {
  args: CliArgs;
}

interface Stats {
  turnCount: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  costUsd?: number;
  contextLimit?: number;
  remainingContext?: number;
  usedContext?: number;
}

interface ReadyInfo {
  model: string;
  provider: string;
  hasProfile: boolean;
}

interface PermissionRequest {
  toolName: string;
  description: string;
}

interface ProviderWizardState {
  step: ProviderStep;
  profiles: ProviderProfile[];
  models: string[];
  fetchError?: string;
  // Local state that accumulates across the wizard's steps (mirrored so the
  // App can call the correct RPC when the user confirms the final step).
  name: string;
  baseUrl: string;
  provider: string;
  apiKey: string;
}

export const App: React.FC<AppProps> = ({ args }) => {
  const { exit } = useApp();
  const [cells, dispatch] = useReducer(cellsReducer, []);
  const [streaming, setStreaming] = useState(false);
  const [ready, setReady] = useState<ReadyInfo | null>(null);
  const [permission, setPermission] = useState<PermissionRequest | null>(null);
  const [question, setQuestion] = useState<string | null>(null);
  const [skills, setSkills] = useState<string[]>([]);
  const [stats, setStats] = useState<Stats>({
    turnCount: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
  });
  const [wizard, setWizard] = useState<ProviderWizardState | null>(null);
  const wizardRef = useRef(wizard);
  wizardRef.current = wizard;

  const spinnerFrame = useSpinner(streaming);

  const handleEvent = (event: Event) => {
    switch (event.event) {
      case "ready":
        setReady({
          model: event.data.model,
          provider: event.data.provider,
          hasProfile: event.data.has_profile,
        });
        setSkills(event.data.skills || []);
        if (!event.data.has_profile) {
          dispatch({
            type: "push",
            cell: {
              kind: "system",
              id: "",
              text:
                "No provider configured. Run /provider to set up a provider profile.",
            },
          });
        }
        break;
      case "text_chunk":
        dispatch({ type: "append_assistant", text: event.data.text });
        break;
      case "thinking_chunk":
        dispatch({ type: "append_thinking", text: event.data.text });
        break;
      case "tool_start":
        dispatch({
          type: "push",
          cell: {
            kind: "tool",
            id: "",
            name: event.data.name,
            args: event.data.inputs || {},
          },
        });
        break;
      case "tool_end":
        dispatch({
          type: "tool_end",
          name: event.data.name,
          result: event.data.result,
          durationMs: event.data.duration_ms,
          permitted: event.data.permitted,
        });
        break;
      case "permission_request":
        setPermission({
          toolName: event.data.tool_name,
          description: event.data.description,
        });
        break;
      case "question_request":
        setQuestion(event.data.question);
        break;
      case "turn_done":
        dispatch({ type: "finalize_assistant" });
        setStreaming(false);
        setStats((s) => ({
          turnCount: s.turnCount + 1,
          totalInputTokens: s.totalInputTokens + event.data.input_tokens,
          totalOutputTokens: s.totalOutputTokens + event.data.output_tokens,
          costUsd: s.costUsd,
        }));
        break;
      case "state":
        setStats({
          turnCount: event.data.turn_count,
          totalInputTokens: event.data.total_input_tokens,
          totalOutputTokens: event.data.total_output_tokens,
          contextLimit: event.data.context_limit,
          remainingContext: event.data.remaining_context,
          usedContext: event.data.used_context,
          costUsd: event.data.cost_usd,
        });
        break;
      case "slash_result":
        dispatch({
          type: "push",
          cell: { kind: "system", id: "", text: event.data.output },
        });
        break;
      case "provider_list":
        setWizard((w) =>
          w
            ? { ...w, profiles: event.data.profiles, step: "select_profile" }
            : null,
        );
        break;
      case "models_list":
        setWizard((w) =>
          w ? { ...w, models: event.data.models, step: "select_model" } : null,
        );
        break;
      case "provider_saved":
        setReady((r) =>
          r
            ? {
                ...r,
                model: event.data.profile.model,
                provider: event.data.profile.provider || r.provider,
                hasProfile: true,
              }
            : r,
        );
        dispatch({
          type: "push",
          cell: { kind: "system", id: "", text: event.data.message },
        });
        setWizard(null);
        break;
      case "model_changed":
        setReady((r) =>
          r
            ? {
                ...r,
                model: event.data.model,
                provider: event.data.provider || r.provider,
              }
            : r,
        );
        break;
      case "skills_updated":
        setSkills(event.data.skills || []);
        break;
      case "query_done":
        setStreaming(false);
        break;
      case "exit":
        exit();
        setTimeout(() => process.exit(0), 0);
        break;
      case "agent_spawn":
        dispatch({
          type: "push",
          cell: {
            kind: "subagent",
            id: "",
            name: event.data.agent_name || event.data.task_id || "agent",
            status: "running",
            text: "",
            streaming: true,
          },
        });
        break;
      case "agent_text":
        dispatch({
          type: "append_subagent_text",
          name: event.data.agent_name || event.data.task_id || "agent",
          text: event.data.text,
        });
        break;
      case "agent_thinking":
        dispatch({
          type: "append_subagent_text",
          name: event.data.agent_name || event.data.task_id || "agent",
          text: `[thinking] ${event.data.text}`,
        });
        break;
      case "agent_tool_start":
        dispatch({
          type: "append_subagent_text",
          name: event.data.agent_name || event.data.task_id || "agent",
          text: `\n◦ ${event.data.tool_name}…`,
        });
        break;
      case "agent_tool_end":
        dispatch({
          type: "append_subagent_text",
          name: event.data.agent_name || event.data.task_id || "agent",
          text: `\n  ${event.data.permitted === false ? "✗" : "✓"} ${event.data.tool_name}${event.data.duration_ms !== undefined ? ` (${event.data.duration_ms.toFixed(0)}ms)` : ""}`,
        });
        break;
      case "agent_done":
        dispatch({
          type: "finalize_subagent",
          name: event.data.agent_name || event.data.task_id || "agent",
          status:
            event.data.status === "failed"
              ? "failed"
              : event.data.status === "cancelled"
                ? "cancelled"
                : "completed",
        });
        break;
      case "error":
        if (wizardRef.current && wizardRef.current.step === "fetching") {
          setWizard({ ...wizardRef.current, fetchError: event.data.message });
          break;
        }
        dispatch({
          type: "push",
          cell: { kind: "error", id: "", text: event.data.message },
        });
        setStreaming(false);
        break;
    }
  };

  const { send } = useBridge({
    python: args.python,
    projectDir: args.projectDir,
    onEvent: handleEvent,
  });

  // Send init once on mount.
  useEffect(() => {
    send({
      method: "init",
      params: {
        model: args.model,
        permission_mode: args.permissionMode,
        base_url: args.baseUrl,
        api_key: args.apiKey,
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const submit = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    if (trimmed.startsWith("/")) {
      const head = trimmed.slice(1).split(/\s+/)[0]?.toLowerCase() ?? "";
      if (head === "exit" || head === "quit" || head === "q") {
        send({ method: "shutdown", params: {} });
        exit();
        return;
      }
      if (head === "clear") {
        dispatch({ type: "clear" });
        return;
      }
      if (head === "provider") {
        // Client-side wizard. The bridge exposes provider_* RPCs, not a /provider slash.
        dispatch({
          type: "push",
          cell: { kind: "user", id: "", text: trimmed },
        });
        setWizard({
          step: "select_profile",
          profiles: [],
          models: [],
          name: "",
          baseUrl: "",
          provider: "",
          apiKey: "",
        });
        send({ method: "provider_list", params: {} });
        return;
      }
      dispatch({
        type: "push",
        cell: { kind: "user", id: "", text: trimmed },
      });
      // Python bridge expects the full `/command` string including the leading slash.
      send({ method: "slash", params: { command: trimmed } });
      return;
    }
    dispatch({
      type: "push",
      cell: { kind: "user", id: "", text: trimmed },
    });
    setStreaming(true);
    send({ method: "query", params: { text: trimmed } });
  };

  const interrupt = () => {
    if (streaming) {
      send({ method: "cancel" });
    } else {
      exit();
    }
  };

  // Global Ctrl+D exits.
  useInput((input, key) => {
    if (key.ctrl && input === "d" && !streaming) exit();
  });

  return (
    <Box flexDirection="column" paddingX={1}>
      {ready && (
        <Header
          version="0.2.0"
          model={ready.model}
          provider={ready.provider}
          cwd={args.projectDir}
          hasProfile={ready.hasProfile}
        />
      )}
      <Cells cells={cells} />
      {question ? (
        <QuestionPrompt
          question={question}
          onResolve={(answer) => {
            send({ method: "question_response", params: { answer } });
            setQuestion(null);
          }}
        />
      ) : permission ? (
        <PermissionPrompt
          toolName={permission.toolName}
          description={permission.description}
          onResolve={(granted) => {
            send({ method: "permission_response", params: { granted } });
            setPermission(null);
          }}
        />
      ) : wizard ? (
        <ProviderDialog
          step={wizard.step}
          profiles={wizard.profiles}
          models={wizard.models}
          fetchError={wizard.fetchError}
          name={wizard.name}
          baseUrl={wizard.baseUrl}
          provider={wizard.provider}
          apiKey={wizard.apiKey}
          send={send}
          onCancel={() => setWizard(null)}
          onDone={(value) => {
            // Advance the wizard one step based on the current step.
            setWizard((w) => {
              if (!w) return null;
              switch (w.step) {
                case "select_profile":
                  return { ...w, step: "enter_name" };
                case "enter_name":
                  return { ...w, name: value || "", step: "enter_provider" };
                case "enter_provider":
                  return { ...w, provider: value || "", step: "enter_url" };
                case "enter_url":
                  return { ...w, baseUrl: value || "", step: "enter_key" };
                case "enter_key": {
                  const updated = { ...w, apiKey: value || "" };
                  send({
                    method: "fetch_models",
                    params: {
                      base_url: updated.baseUrl || "",
                      api_key: updated.apiKey || "",
                    },
                  });
                  return { ...updated, step: "fetching" };
                }
                case "select_model":
                  return w;
                case "fetching":
                  return { ...w, fetchError: undefined };
                default:
                  return w;
              }
            });
          }}
        />
      ) : (
        <Composer
          disabled={streaming}
          skills={skills}
          onSubmit={submit}
          onInterrupt={interrupt}
        />
      )}
      <Footer
        turnCount={stats.turnCount}
        inputTokens={stats.totalInputTokens}
        outputTokens={stats.totalOutputTokens}
        costUsd={stats.costUsd}
        contextLimit={stats.contextLimit}
        remainingContext={stats.remainingContext}
        usedContext={stats.usedContext}
        streaming={streaming}
        spinnerFrame={spinnerFrame}
      />
    </Box>
  );
};
