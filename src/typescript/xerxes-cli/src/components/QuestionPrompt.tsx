import React, { useRef, useState } from "react";
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

  const valueRef = useRef(value);
  const cursorRef = useRef(cursor);
  valueRef.current = value;
  cursorRef.current = cursor;

  useInput(
    (input, key) => {
      if (submitted) return;

      const curVal = valueRef.current;
      const curCursor = cursorRef.current;

      if (key.escape) {
        setSubmitted(true);
        onResolve("[cancelled]");
        return;
      }

      if (key.return) {
        setSubmitted(true);
        onResolve(curVal);
        return;
      }

      if (key.backspace) {
        if (curCursor > 0) {
          const nextVal =
            curVal.slice(0, curCursor - 1) + curVal.slice(curCursor);
          setValue(nextVal);
          setCursor(curCursor - 1);
        }
        return;
      }

      if (key.delete) {
        const nextEnd = Math.min(curVal.length, curCursor + 1);
        if (nextEnd > curCursor) {
          const nextVal = curVal.slice(0, curCursor) + curVal.slice(nextEnd);
          setValue(nextVal);
        }
        return;
      }

      if (key.leftArrow) {
        setCursor(Math.max(0, curCursor - 1));
        return;
      }

      if (key.rightArrow) {
        setCursor(Math.min(curVal.length, curCursor + 1));
        return;
      }

      if (key.home) {
        setCursor(0);
        return;
      }

      if (key.end) {
        setCursor(curVal.length);
        return;
      }

      if (input && !key.ctrl && !key.meta) {
        const nextVal =
          curVal.slice(0, curCursor) + input + curVal.slice(curCursor);
        setValue(nextVal);
        setCursor(curCursor + input.length);
      }
    },
    { isActive: !submitted },
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
