import React, { useRef } from "react";
import { Box, Text, useInput } from "ink";

interface PermissionPromptProps {
  toolName: string;
  description: string;
  onResolve: (granted: boolean) => void;
}

export const PermissionPrompt: React.FC<PermissionPromptProps> = ({
  toolName,
  description,
  onResolve,
}) => {
  const resolved = useRef(false);
  useInput((input, key) => {
    if (resolved.current) return;
    if (key.return || input.toLowerCase() === "y" || input.toLowerCase() === "a") {
      resolved.current = true;
      onResolve(true);
    } else if (key.escape || input.toLowerCase() === "n") {
      resolved.current = true;
      onResolve(false);
    }
  });

  return (
    <Box
      marginTop={1}
      borderStyle="round"
      borderColor="yellow"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color="yellow">
        ⚠ permission required
      </Text>
      <Text>
        <Text bold>{toolName}</Text> wants to run.
      </Text>
      <Text dimColor>{description}</Text>
      <Box marginTop={1}>
        <Text dimColor>[y] approve  [n] deny  [esc] deny</Text>
      </Box>
    </Box>
  );
};
