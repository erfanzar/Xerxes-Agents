import React, { useState } from "react";
import { Box, Text, useInput } from "ink";

interface QuestionPromptProps {
  question: string;
  onResolve: (answer: string) => void;
}

export const QuestionPrompt: React.FC<QuestionPromptProps> = ({
  question,
  onResolve,
}) => {
  const [value, setValue] = useState("");
  const [cursor, setCursor] = useState(0);
  const [submitted, setSubmitted] = useState(false);

  useInput(
    (input, key) => {
      if (submitted) return;

      if (key.escape) {
        setSubmitted(true);
        onResolve("[cancelled]");
        return;
      }

      if (key.return) {
        setSubmitted(true);
        onResolve(value);
        return;
      }

      if (key.backspace || key.delete) {
        if (cursor > 0) {
          const before = value.slice(0, cursor - 1);
          const after = value.slice(cursor);
          setValue(before + after);
          setCursor(cursor - 1);
        }
        return;
      }

      if (key.leftArrow) {
        setCursor(Math.max(0, cursor - 1));
        return;
      }

      if (key.rightArrow) {
        setCursor(Math.min(value.length, cursor + 1));
        return;
      }

      if (key.home) {
        setCursor(0);
        return;
      }

      if (key.end) {
        setCursor(value.length);
        return;
      }

      if (input && !key.ctrl && !key.meta) {
        const before = value.slice(0, cursor);
        const after = value.slice(cursor);
        setValue(before + input + after);
        setCursor(cursor + input.length);
      }
    },
    { isActive: !submitted }
  );

  if (submitted) {
    return (
      <Box marginTop={1} flexDirection="column">
        <Text dimColor>Answer submitted.</Text>
      </Box>
    );
  }

  const beforeCursor = value.slice(0, cursor);
  const atCursor = value[cursor] || " ";
  const afterCursor = value.slice(cursor + 1);

  return (
    <Box
      marginTop={1}
      borderStyle="round"
      borderColor="cyan"
      flexDirection="column"
      paddingX={1}
    >
      <Text bold color="cyan">
        ? question
      </Text>
      <Text>{question}</Text>
      <Box marginTop={1}>
        <Text color="cyan">{"> "}</Text>
        <Text>{beforeCursor}</Text>
        <Text backgroundColor="cyan" color="black">
          {atCursor}
        </Text>
        <Text>{afterCursor}</Text>
      </Box>
      <Box marginTop={1}>
        <Text dimColor>[enter] submit  [esc] cancel</Text>
      </Box>
    </Box>
  );
};
