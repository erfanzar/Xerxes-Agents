import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import type { Request, ProviderProfile } from "../bridge/types.js";

/**
 * Multi-step provider-setup wizard invoked via `/provider`.
 *
 * Flow:
 *   SELECT_PROFILE → (existing) → done / (new) → ENTER_NAME → ENTER_PROVIDER
 *   → ENTER_URL → ENTER_KEY → FETCHING → SELECT_MODEL → done
 */
export type ProviderStep =
  | "select_profile"
  | "enter_name"
  | "enter_provider"
  | "enter_url"
  | "enter_key"
  | "fetching"
  | "select_model";

interface ProviderDialogProps {
  step: ProviderStep;
  profiles: ProviderProfile[];
  models: string[];
  fetchError?: string;
  name: string;
  baseUrl: string;
  provider: string;
  apiKey: string;
  send: (req: Request) => void;
  onDone: (value?: string) => void;
  onCancel: () => void;
}

export const ProviderDialog: React.FC<ProviderDialogProps> = ({
  step,
  profiles,
  models,
  fetchError,
  name,
  baseUrl,
  provider,
  apiKey,
  send,
  onDone,
  onCancel,
}) => {
  const PROVIDER_OPTIONS = [
    "openai",
    "anthropic",
    "kimi",
    "deepseek",
    "together",
    "groq",
    "minimax",
    "ollama",
    "local",
    "custom",
  ];

  const [text, setText] = useState("");
  const [profileIdx, setProfileIdx] = useState(0);
  const [providerIdx, setProviderIdx] = useState(0);
  const [modelIdx, setModelIdx] = useState(0);

  useEffect(() => {
    if (step === "select_model") {
      setModelIdx(0);
    }
  }, [step, models]);

  // Return the canonical base URL for a known provider.
  const defaultUrlForProvider = (p: string): string => {
    switch (p.toLowerCase()) {
      case "openai":
        return "https://api.openai.com/v1";
      case "anthropic":
        return "https://api.anthropic.com/v1";
      case "kimi":
        return "https://api.moonshot.ai/v1";
      case "deepseek":
        return "https://api.deepseek.com/v1";
      case "together":
        return "https://api.together.xyz/v1";
      case "groq":
        return "https://api.groq.com/openai/v1";
      case "minimax":
        return "https://api.minimax.chat/v1";
      case "ollama":
        return "http://localhost:11434/v1";
      case "local":
        return "http://localhost:8000/v1";
      default:
        return "";
    }
  };

  // Reset the typed buffer whenever the step changes.
  // Pre-fill URL when entering the URL step based on the chosen provider.
  useEffect(() => {
    if (step === "enter_url") {
      const defaultUrl = defaultUrlForProvider(provider);
      if (defaultUrl) {
        setText(defaultUrl);
      } else {
        setText("");
      }
    } else {
      setText("");
    }
  }, [step, provider]);

  useInput(
    (input, key) => {
      if (key.escape) {
        onCancel();
        return;
      }

      // --- select_profile ---
      if (step === "select_profile") {
        const options = [...profiles, { name: "+ New profile", __new: true }];
        if (key.upArrow) {
          setProfileIdx((i) => Math.max(0, i - 1));
        } else if (key.downArrow) {
          setProfileIdx((i) => Math.min(options.length - 1, i + 1));
        } else if (key.return) {
          const pick = options[profileIdx];
          if (!pick) return;
          if ((pick as { __new?: boolean }).__new) {
            onDone();
          } else {
            send({
              method: "provider_select",
              params: { name: (pick as ProviderProfile).name },
            });
            // Bridge will emit provider_saved → parent closes wizard.
          }
        }
        return;
      }

      // --- select_provider ---
      if (step === "enter_provider") {
        if (key.upArrow) {
          setProviderIdx((i) => Math.max(0, i - 1));
        } else if (key.downArrow) {
          setProviderIdx((i) =>
            Math.min(PROVIDER_OPTIONS.length - 1, i + 1),
          );
        } else if (key.return) {
          const picked = PROVIDER_OPTIONS[providerIdx];
          if (picked) onDone(picked);
        }
        return;
      }

      // --- text-entry steps ---
      if (
        step === "enter_name" ||
        step === "enter_url" ||
        step === "enter_key"
      ) {
        if (key.return) {
          onDone(text);
          return;
        }
        if (key.backspace || key.delete) {
          setText((t) => t.slice(0, -1));
          return;
        }
        if (input && !key.ctrl && !key.meta) {
          setText((t) => t + input);
        }
        return;
      }

      // --- fetching (retry on error) ---
      if (step === "fetching") {
        if (key.escape) {
          onCancel();
          return;
        }
        if (fetchError && key.return) {
          send({
            method: "fetch_models",
            params: { base_url: baseUrl, api_key: apiKey },
          });
          onDone();
          return;
        }
        return;
      }

      // --- select_model ---
      if (step === "select_model") {
        if (key.upArrow) {
          setModelIdx((i) => Math.max(0, i - 1));
        } else if (key.downArrow) {
          setModelIdx((i) => Math.min(models.length - 1, i + 1));
        } else if (key.return) {
          const picked = models[modelIdx];
          if (picked) {
            send({
              method: "provider_save",
              params: {
                name: name || "default",
                base_url: baseUrl,
                api_key: apiKey,
                model: picked,
                provider: provider || "custom",
              },
            });
            onDone();
          }
        }
        return;
      }
    },
    { isActive: true },
  );

  // --- render ---
  return (
    <Box
      marginTop={1}
      borderStyle="round"
      borderColor="magenta"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color="magenta">
        ⚙ provider setup
      </Text>
      {step === "select_profile" && (
        <>
          <Text dimColor>Select a profile or create a new one:</Text>
          <Box flexDirection="column" marginTop={1}>
            {profiles.map((p, i) => (
              <Text
                key={p.name}
                color={i === profileIdx ? "magenta" : undefined}
                bold={i === profileIdx}
              >
                {i === profileIdx ? "› " : "  "}
                {p.active ? "* " : "  "}
                {p.name}
                <Text dimColor>{`  ${p.model}  ${p.base_url}`}</Text>
              </Text>
            ))}
            <Text
              color={profileIdx === profiles.length ? "magenta" : undefined}
              bold={profileIdx === profiles.length}
            >
              {profileIdx === profiles.length ? "› " : "  "}
              {"   + New profile"}
            </Text>
          </Box>
        </>
      )}

      {step === "enter_name" && (
        <>
          <Text dimColor>Profile name:</Text>
          <Text>
            <Text color="magenta">› </Text>
            {text}
            <Text inverse> </Text>
          </Text>
        </>
      )}

      {step === "enter_provider" && (
        <>
          <Text dimColor>Select a provider:</Text>
          <Box flexDirection="column" marginTop={1}>
            {PROVIDER_OPTIONS.map((p, i) => (
              <Text
                key={p}
                color={i === providerIdx ? "magenta" : undefined}
                bold={i === providerIdx}
              >
                {i === providerIdx ? "› " : "  "}
                {p}
              </Text>
            ))}
          </Box>
        </>
      )}

      {step === "enter_url" && (
        <>
          <Text dimColor>
            Base URL (press Enter to accept default):
          </Text>
          <Text>
            <Text color="magenta">› </Text>
            {text}
            <Text inverse> </Text>
          </Text>
        </>
      )}

      {step === "enter_key" && (
        <>
          <Text dimColor>API key (leave blank for local / no auth):</Text>
          <Text>
            <Text color="magenta">› </Text>
            {"•".repeat(text.length)}
            <Text inverse> </Text>
          </Text>
        </>
      )}

      {step === "fetching" && (
        <>
          <Text color="yellow">⏳ Fetching models from {baseUrl}…</Text>
          {fetchError && (
            <Text color="red" bold>
              {fetchError}
            </Text>
          )}
        </>
      )}

      {step === "select_model" && (
        <>
          <Text dimColor>Select a model:</Text>
          <Box flexDirection="column" marginTop={1}>
            {models.length === 0 ? (
              <Text dimColor>No models available</Text>
            ) : (
              (() => {
                const maxRows = 11;
                const halfRows = Math.floor(maxRows / 2);
                const start = Math.max(0, Math.min(models.length - maxRows, modelIdx - halfRows));
                return models.slice(start, start + maxRows).map((m, i) => {
                  const actualIdx = start + i;
                  return (
                    <Text
                      key={m}
                      color={actualIdx === modelIdx ? "magenta" : undefined}
                      bold={actualIdx === modelIdx}
                    >
                      {actualIdx === modelIdx ? "› " : "  "}
                      {m}
                    </Text>
                  );
                });
              })()
            )}
            {models.length > 0 && (
              <Text dimColor>
                {"  "}
                {Math.min(modelIdx + 1, models.length)} / {models.length}
              </Text>
            )}
          </Box>
        </>
      )}

      <Box marginTop={1}>
        <Text dimColor>[↑↓ to navigate · enter to confirm · esc to cancel]</Text>
      </Box>
    </Box>
  );
};
