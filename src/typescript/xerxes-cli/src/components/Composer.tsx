import React, { useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { SlashMenu, matchCommands } from "./SlashMenu.js";

interface ComposerProps {
  disabled: boolean;
  placeholder?: string;
  skills?: string[];
  onSubmit: (text: string) => void;
  onInterrupt?: () => void;
}

interface ComposerState {
  value: string;
  cursor: number;
  slashIndex: number;
}

const initialComposerState = (): ComposerState => ({
  value: "",
  cursor: 0,
  slashIndex: 0,
});

const findPreviousWordStart = (value: string, cursor: number): number => {
  let index = cursor;
  while (index > 0 && /\s/.test(value[index - 1] ?? "")) {
    index -= 1;
  }
  while (index > 0 && !/\s/.test(value[index - 1] ?? "")) {
    index -= 1;
  }
  return index;
};

const findNextWordEnd = (value: string, cursor: number): number => {
  let index = cursor;
  while (index < value.length && /\s/.test(value[index] ?? "")) {
    index += 1;
  }
  while (index < value.length && !/\s/.test(value[index] ?? "")) {
    index += 1;
  }
  return index;
};

export const Composer: React.FC<ComposerProps> = ({
  disabled,
  placeholder = "Type a message, / for commands, Ctrl+C to exit",
  skills = [],
  onSubmit,
  onInterrupt,
}) => {
  const [state, setState] = useState<ComposerState>(() => initialComposerState());
  const stateRef = useRef(state);
  stateRef.current = state;

  // When the buffer starts with `/` and has no spaces, show the slash menu.
  const { value, cursor, slashIndex } = state;
  const showSlash =
    !disabled && value.startsWith("/") && !value.includes(" ");
  const slashQuery = showSlash ? value.slice(1) : "";
  const slashMatches = showSlash ? matchCommands(slashQuery, skills) : [];

  useInput(
    (input, key) => {
      if (disabled) {
        if (key.ctrl && input === "c" && onInterrupt) {
          onInterrupt();
        }
        return;
      }

      const current = stateRef.current;
      const showSlashNow =
        current.value.startsWith("/") && !current.value.includes(" ");
      const slashMatchesNow = showSlashNow
        ? matchCommands(current.value.slice(1), skills)
        : [];

      // Slash menu navigation takes priority.
      if (showSlashNow && slashMatchesNow.length > 0) {
        if (key.upArrow) {
          setState((prev) => ({
            ...prev,
            slashIndex: Math.max(0, prev.slashIndex - 1),
          }));
          return;
        }
        if (key.downArrow) {
          setState((prev) => ({
            ...prev,
            slashIndex: Math.min(slashMatchesNow.length - 1, prev.slashIndex + 1),
          }));
          return;
        }
        if (key.tab) {
          // Complete to the currently-selected command.
          const pick = slashMatchesNow[current.slashIndex];
          if (pick) {
            const nextValue = "/" + pick.name;
            setState({
              value: nextValue,
              cursor: nextValue.length,
              slashIndex: 0,
            });
          }
          return;
        }
        if (key.return) {
          const pick = slashMatchesNow[current.slashIndex];
          if (pick) {
            onSubmit("/" + pick.name);
            setState(initialComposerState());
          }
          return;
        }
        if (key.escape) {
          setState(initialComposerState());
          return;
        }
      }

      if (key.return) {
        if (current.value.trim().length > 0) {
          onSubmit(current.value);
          setState(initialComposerState());
        }
        return;
      }
      if (key.backspace) {
        const deleteWord = key.ctrl || key.meta;
        setState((prev) => {
          if (prev.cursor === 0) return prev;
          const nextCursor = deleteWord
            ? findPreviousWordStart(prev.value, prev.cursor)
            : prev.cursor - 1;
          return {
            value: prev.value.slice(0, nextCursor) + prev.value.slice(prev.cursor),
            cursor: nextCursor,
            slashIndex: 0,
          };
        });
        return;
      }
      if (key.delete) {
        const deleteWord = key.ctrl || key.meta;
        setState((prev) => {
          const nextCursor = deleteWord
            ? findNextWordEnd(prev.value, prev.cursor)
            : Math.min(prev.value.length, prev.cursor + 1);
          if (nextCursor === prev.cursor) {
            // Some terminals send key.delete for Backspace. When the cursor
            // is at the end of the text, forward-delete is a no-op — fall
            // back to backspace behaviour in that case.
            if (prev.cursor === 0) return prev;
            const backCursor = deleteWord
              ? findPreviousWordStart(prev.value, prev.cursor)
              : prev.cursor - 1;
            return {
              value: prev.value.slice(0, backCursor) + prev.value.slice(prev.cursor),
              cursor: backCursor,
              slashIndex: 0,
            };
          }
          return {
            value: prev.value.slice(0, prev.cursor) + prev.value.slice(nextCursor),
            cursor: prev.cursor,
            slashIndex: 0,
          };
        });
        return;
      }
      if (key.leftArrow) {
        setState((prev) => ({
          ...prev,
          cursor: Math.max(0, prev.cursor - 1),
        }));
        return;
      }
      if (key.rightArrow) {
        setState((prev) => ({
          ...prev,
          cursor: Math.min(prev.value.length, prev.cursor + 1),
        }));
        return;
      }
      if (key.ctrl && input === "a") {
        setState((prev) => ({ ...prev, cursor: 0 }));
        return;
      }
      if (key.ctrl && input === "e") {
        setState((prev) => ({ ...prev, cursor: prev.value.length }));
        return;
      }
      if (key.ctrl && input === "u") {
        setState((prev) => ({
          value: prev.value.slice(prev.cursor),
          cursor: 0,
          slashIndex: 0,
        }));
        return;
      }
      if (key.ctrl && input === "k") {
        setState((prev) => ({
          ...prev,
          value: prev.value.slice(0, prev.cursor),
        }));
        return;
      }
      if (key.ctrl && input === "w") {
        setState((prev) => {
          const nextCursor = findPreviousWordStart(prev.value, prev.cursor);
          return {
            value: prev.value.slice(0, nextCursor) + prev.value.slice(prev.cursor),
            cursor: nextCursor,
            slashIndex: 0,
          };
        });
        return;
      }
      if (key.ctrl && input === "c") {
        if (onInterrupt) {
          onInterrupt();
        }
        setState(initialComposerState());
        return;
      }
      // Regular input
      if (input && !key.ctrl && !key.meta) {
        setState((prev) => ({
          value: prev.value.slice(0, prev.cursor) + input + prev.value.slice(prev.cursor),
          cursor: prev.cursor + input.length,
          slashIndex: 0,
        }));
      }
    },
    { isActive: true },
  );

  const before = value.slice(0, cursor);
  const at = value.slice(cursor, cursor + 1);
  const after = value.slice(cursor + 1);
  const showPlaceholder = value.length === 0 && !disabled;

  return (
    <Box marginTop={1} flexDirection="column">
      <Box>
        <Text bold color={disabled ? "yellow" : "magenta"}>
          {disabled ? "… " : "› "}
        </Text>
        {showPlaceholder ? (
          <Text dimColor>{placeholder}</Text>
        ) : (
          <>
            <Text>{before}</Text>
            <Text inverse>{at || " "}</Text>
            <Text>{after}</Text>
          </>
        )}
      </Box>
      {showSlash && (
        <SlashMenu query={slashQuery} selectedIndex={slashIndex} skills={skills} />
      )}
    </Box>
  );
};
