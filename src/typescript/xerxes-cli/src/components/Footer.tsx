import React from "react";
import { Box, Text } from "ink";

interface FooterProps {
  turnCount: number;
  inputTokens: number;
  outputTokens: number;
  costUsd?: number;
  contextLimit?: number;
  remainingContext?: number;
  streaming: boolean;
  spinnerFrame: string;
}

export const Footer: React.FC<FooterProps> = ({
  turnCount,
  inputTokens,
  outputTokens,
  costUsd,
  contextLimit,
  remainingContext,
  streaming,
  spinnerFrame,
}) => {
  const ctxText =
    contextLimit && remainingContext !== undefined
      ? `${fmt(remainingContext)} / ${fmt(contextLimit)} ctx`
      : "";
  const ctxColor =
    remainingContext !== undefined && remainingContext < 10_000
      ? "red"
      : remainingContext !== undefined && remainingContext < 30_000
        ? "yellow"
        : "dimColor";

  return (
    <Box marginTop={1} justifyContent="space-between">
      <Box>
        {streaming ? (
          <>
            <Text color="yellow">{spinnerFrame} </Text>
            <Text dimColor>Working… (Ctrl+C to interrupt)</Text>
          </>
        ) : (
          <Text dimColor>Ready. Type a message or /help</Text>
        )}
      </Box>
      <Box>
        <Text dimColor>
          {`turns ${turnCount} · in ${fmt(inputTokens)} · out ${fmt(outputTokens)}`}
          {costUsd !== undefined ? ` · $${costUsd.toFixed(4)}` : ""}
        </Text>
        {ctxText && (
          <Text color={ctxColor}> · {ctxText}</Text>
        )}
      </Box>
    </Box>
  );
};

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}
