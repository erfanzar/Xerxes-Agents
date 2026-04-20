import React, { useState } from "react";
import { Box, Text, useInput, useApp, render } from "ink";

function DebugApp() {
  const { exit } = useApp();
  const [lines, setLines] = useState<string[]>([]);
  useInput((input, key) => {
    const flags = Object.entries(key)
      .filter(([_, v]) => v === true)
      .map(([k]) => k)
      .join(", ");
    const codes = [...input].map((c) => `0x${c.charCodeAt(0).toString(16)}`);
    const line = `input="${input}" codes=[${codes.join(", ")}] flags={${flags}}`;
    setLines((prev) => [...prev.slice(-14), line]);
    if (key.escape && input === "") exit();
  });
  return (
    <Box flexDirection="column">
      <Text bold color="green">Press keys. Esc to exit.</Text>
      {lines.map((l, i) => (
        <Text key={i}>{l}</Text>
      ))}
    </Box>
  );
}

render(<DebugApp />);
