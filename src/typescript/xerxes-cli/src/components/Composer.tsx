import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { SlashMenu, matchCommands } from "./SlashMenu.js";

interface ComposerProps {
  disabled: boolean;
  placeholder?: string;
  skills?: string[];
  onSubmit: (text: string) => void;
  onInterrupt?: () => void;
}

export const Composer: React.FC<ComposerProps> = ({
  disabled,
  placeholder = "Type a message, / for commands, Ctrl+C to exit",
  skills = [],
  onSubmit,
  onInterrupt,
}) => {
  const [value, setValue] = useState("");
  const [cursor, setCursor] = useState(0);
  const [slashIndex, setSlashIndex] = useState(0);

  // When the buffer starts with `/` and has no spaces, show the slash menu.
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

      // Slash menu navigation takes priority.
      if (showSlash && slashMatches.length > 0) {
        if (key.upArrow) {
          setSlashIndex((i) => Math.max(0, i - 1));
          return;
        }
        if (key.downArrow) {
          setSlashIndex((i) => Math.min(slashMatches.length - 1, i + 1));
          return;
        }
        if (key.tab) {
          // Complete to the currently-selected command.
          const pick = slashMatches[slashIndex];
          if (pick) {
            setValue("/" + pick.name);
            setCursor(("/" + pick.name).length);
            setSlashIndex(0);
          }
          return;
        }
        if (key.return) {
          const pick = slashMatches[slashIndex];
          if (pick) {
            onSubmit("/" + pick.name);
            setValue("");
            setCursor(0);
            setSlashIndex(0);
          }
          return;
        }
        if (key.escape) {
          setValue("");
          setCursor(0);
          setSlashIndex(0);
          return;
        }
      }

      if (key.return) {
        if (value.trim().length > 0) {
          onSubmit(value);
          setValue("");
          setCursor(0);
          setSlashIndex(0);
        }
        return;
      }
      if (key.backspace || key.delete) {
        if (cursor > 0) {
          const next = value.slice(0, cursor - 1) + value.slice(cursor);
          setValue(next);
          setCursor(cursor - 1);
          setSlashIndex(0);
        }
        // At cursor 0: nothing to delete; explicitly ignore rather than silently return.
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
      if (key.ctrl && input === "a") {
        setCursor(0);
        return;
      }
      if (key.ctrl && input === "e") {
        setCursor(value.length);
        return;
      }
      if (key.ctrl && input === "u") {
        setValue(value.slice(cursor));
        setCursor(0);
        setSlashIndex(0);
        return;
      }
      if (key.ctrl && input === "k") {
        setValue(value.slice(0, cursor));
        return;
      }
      if (key.ctrl && input === "w") {
        const before = value.slice(0, cursor);
        const stripped = before.replace(/\s*\S+\s*$/, "");
        setValue(stripped + value.slice(cursor));
        setCursor(stripped.length);
        setSlashIndex(0);
        return;
      }
      if (key.ctrl && input === "c") {
        if (onInterrupt) {
          onInterrupt();
        }
        setValue("");
        setCursor(0);
        setSlashIndex(0);
        return;
      }
      // Regular input
      if (input && !key.ctrl && !key.meta) {
        const next = value.slice(0, cursor) + input + value.slice(cursor);
        setValue(next);
        setCursor(cursor + input.length);
        setSlashIndex(0);
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
