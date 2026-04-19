import React from "react";
import { Box, Text } from "ink";

interface FooterProps {
  turnCount: number;
  inputTokens: number;
  outputTokens: number;
  costUsd?: number;
  streaming: boolean;
  spinnerFrame: string;
}

export const Footer: React.FC<FooterProps> = ({
  turnCount,
  inputTokens,
  outputTokens,
  costUsd,
  streaming,
  spinnerFrame,
}) => (
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
    </Box>
  </Box>
);

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}
