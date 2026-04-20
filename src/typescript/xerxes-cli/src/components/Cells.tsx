import React from "react";
import { Box, Text } from "ink";
import { MarkdownText } from "./Markdown.js";
import { useSpinner } from "../hooks/useSpinner.js";
import type { Cell } from "../state/cells.js";

const UserCell: React.FC<{ text: string }> = ({ text }) => (
  <Box>
    <Text bold dimColor>
      {"› "}
    </Text>
    <Text>{text}</Text>
  </Box>
);

function normalizeText(text: string): string {
  // Trim trailing whitespace, fully collapse consecutive newlines to 1,
  // trim each line's trailing spaces, and strip leading/trailing empty lines.
  let result = text
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n");
  // Iteratively collapse so \n\n\n → \n (regex /g alone leaves overlaps).
  while (result.includes("\n\n")) {
    result = result.replaceAll("\n\n", "\n");
  }
  return result.trim().replace(/^\n+/, "");
}

const AssistantCell: React.FC<{ text: string; streaming: boolean }> = ({
  text,
  streaming,
}) => {
  return (
    <Box flexDirection="column">
      <MarkdownText text={normalizeText(text)} streaming={streaming} />
    </Box>
  );
};

const ThinkingCell: React.FC<{ text: string }> = ({ text }) => (
  <Box flexDirection="column">
    <Text dimColor italic>{normalizeText(text)}</Text>
  </Box>
);

const ToolCell: React.FC<{
  name: string;
  args: Record<string, unknown>;
  result?: string;
  durationMs?: number;
  permitted?: boolean;
}> = ({ name, args, result, durationMs, permitted }) => {
  const pending = result === undefined;
  const spinnerFrame = useSpinner(pending);
  const status = pending
    ? `${spinnerFrame} `
    : permitted === false
      ? "✗ "
      : "✓ ";
  const statusColor =
    pending ? "yellow" : permitted === false ? "red" : "green";
  const argPreview = formatArgs(args);
  const resultLines = result ? result.split("\n").slice(0, 5) : [];
  const truncated = result ? result.split("\n").length > 5 : false;
  
  // Show permission hint when tool is pending and will be denied
  const showPermissionHint = result === undefined && permitted === false;
  
  return (
    <Box flexDirection="column">
      <Box>
        <Text color={statusColor} bold>
          {status}
        </Text>
        <Text bold>{name}</Text>
        {argPreview ? (
          <>
            <Text> </Text>
            <Text color="cyan">{argPreview}</Text>
          </>
        ) : null}
        {durationMs !== undefined ? (
          <Text dimColor>{`  (${durationMs.toFixed(0)}ms)`}</Text>
        ) : null}
        {showPermissionHint && (
          <Text dimColor>  (will be denied)</Text>
        )}
      </Box>
      {resultLines.length > 0 && (
        <Box flexDirection="column" paddingLeft={2}>
          {resultLines.map((line, i) => (
            <Text key={i} dimColor>
              {i === 0 ? "└ " : "  "}
              {line}
            </Text>
          ))}
          {truncated && <Text dimColor>  … (truncated)</Text>}
        </Box>
      )}
    </Box>
  );
};

const SystemCell: React.FC<{ text: string }> = ({ text }) => (
  <Box>
    <Text dimColor>{text}</Text>
  </Box>
);

const ErrorCell: React.FC<{ text: string }> = ({ text }) => (
  <Box>
    <Text color="red" bold>
      ✗{" "}
    </Text>
    <Text color="red">{text}</Text>
  </Box>
);

const SubAgentCell: React.FC<{
  name: string;
  text: string;
  streaming: boolean;
  status: "running" | "completed" | "failed" | "cancelled";
}> = ({ name, text, streaming, status }) => {
  const statusIcon =
    status === "running" ? "◐" : status === "failed" ? "✗" : status === "cancelled" ? "⊘" : "✓";
  const statusColor =
    status === "running" ? "yellow" : status === "failed" ? "red" : status === "cancelled" ? "gray" : "green";
  const lines = normalizeText(text).split("\n").filter((l) => l.trim());
  return (
    <Box flexDirection="column">
      <Box>
        <Text color={statusColor} bold>
          {statusIcon}{" "}
        </Text>
        <Text bold color="cyan">
          {name}
        </Text>
        <Text dimColor> (sub-agent)</Text>
        {streaming && <Text dimColor> ▋</Text>}
      </Box>
      {lines.length > 0 && (
        <Box flexDirection="column" paddingLeft={2}>
          {lines.map((line, i) => (
            <Text key={i} dimColor>
              {i === 0 ? "└ " : "  "}
              {line}
            </Text>
          ))}
        </Box>
      )}
    </Box>
  );
};

function formatArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  const parts = entries
    .slice(0, 3)
    .map(([k, v]) => `${k}=${shortValue(v)}`)
    .join(" ");
  return entries.length > 3 ? `${parts} …` : parts;
}

function shortValue(v: unknown): string {
  if (typeof v === "string") {
    return v.length > 40 ? v.slice(0, 37) + "…" : v;
  }
  return JSON.stringify(v);
}

export const Cells: React.FC<{ cells: Cell[] }> = ({ cells }) => (
  <Box flexDirection="column">
    {cells.map((c) => {
      switch (c.kind) {
        case "user":
          return <UserCell key={c.id} text={c.text} />;
        case "assistant":
          return (
            <AssistantCell key={c.id} text={c.text} streaming={c.streaming} />
          );
        case "thinking":
          return <ThinkingCell key={c.id} text={c.text} />;
        case "tool":
          return (
            <ToolCell
              key={c.id}
              name={c.name}
              args={c.args}
              result={c.result}
              durationMs={c.durationMs}
              permitted={c.permitted}
            />
          );
        case "system":
          return <SystemCell key={c.id} text={c.text} />;
        case "error":
          return <ErrorCell key={c.id} text={c.text} />;
        case "subagent":
          return (
            <SubAgentCell
              key={c.id}
              name={c.name}
              text={c.text}
              streaming={c.streaming}
              status={c.status}
            />
          );
      }
    })}
  </Box>
);
