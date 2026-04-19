import React, { useMemo } from "react";
import { Box, Text } from "ink";
import Gradient from "ink-gradient";
import figlet from "figlet";
import ansiShadow from "figlet/importable-fonts/ANSI Shadow.js";
import { tildify } from "../utils/cwd.js";

figlet.parseFont("ANSI Shadow", ansiShadow);

const PERSIAN_ROYAL = ["#9333EA", "#C084FC", "#FDE68A", "#C9A227"];

interface HeaderProps {
  version: string;
  model: string;
  provider: string;
  cwd: string;
  hasProfile: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  version,
  model,
  provider,
  cwd,
  hasProfile,
}) => {
  const logoLines = useMemo(() => {
    const raw = figlet.textSync("XERXES", { font: "ANSI Shadow" });
    return raw.split("\n").filter((l) => l.trim().length > 0);
  }, []);

  const modelLine = hasProfile
    ? `model:  ${model}${provider ? ` (${provider})` : ""}`
    : "model:  (none)";
  const dirLine = `dir:    ${tildify(cwd)}`;
  const versionLine = `>_ Xerxes (v${version})`;

  const statusLines = [versionLine, modelLine, dirLine];
  const statusWidth = Math.max(...statusLines.map((l) => l.length), 30);

  const statusBorder = "─".repeat(statusWidth + 2);
  const statusPad = (s: string) => s.padEnd(statusWidth);

  return (
    <Box flexDirection="row" marginY={1}>
      {/* LEFT: Raw gradient logo, no box */}
      <Box flexDirection="column">
        {logoLines.map((line, i) => (
          <Gradient key={`logo-${i}`} colors={PERSIAN_ROYAL}>
            <Text>{line}</Text>
          </Gradient>
        ))}
      </Box>

      {/* Gap between columns */}
      <Box marginX={2} />

      {/* RIGHT: Status box */}
      <Box flexDirection="column">
        <Text dimColor>{`┌${statusBorder}┐`}</Text>
        <Box flexDirection="row">
          <Text dimColor>{"│ "}</Text>
          <Text bold color="cyan">{statusPad(versionLine)}</Text>
          <Text dimColor>{" │"}</Text>
        </Box>
        <Text dimColor>{`├${statusBorder}┤`}</Text>
        <Box flexDirection="row">
          <Text dimColor>{"│ "}</Text>
          <Text color="white">{statusPad(modelLine)}</Text>
          <Text dimColor>{" │"}</Text>
        </Box>
        <Box flexDirection="row">
          <Text dimColor>{"│ "}</Text>
          <Text dimColor>{statusPad(dirLine)}</Text>
          <Text dimColor>{" │"}</Text>
        </Box>
        <Text dimColor>{`└${statusBorder}┘`}</Text>
      </Box>
    </Box>
  );
};
