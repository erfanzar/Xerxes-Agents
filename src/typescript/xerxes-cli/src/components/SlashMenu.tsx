import React from "react";
import { Box, Text } from "ink";

export interface SlashCommand {
  name: string;
  description: string;
}

// Full command catalogue the Python bridge understands.
// Kept in sync with `src/python/xerxes/bridge/server.py::handle_slash`.
export const SLASH_COMMANDS: SlashCommand[] = [
  { name: "help", description: "Show available commands" },
  { name: "provider", description: "Setup or switch provider profile" },
  { name: "model", description: "Switch model" },
  { name: "sampling", description: "View / set temperature, top_p, max_tokens" },
  { name: "cost", description: "Show token usage and USD cost" },
  { name: "context", description: "Show session context usage" },
  { name: "compact", description: "Summarize conversation to free context" },
  { name: "clear", description: "Clear conversation" },
  { name: "tools", description: "List active tools" },
  { name: "skills", description: "List available skills" },
  { name: "skill", description: "Invoke a skill by name" },
  { name: "skill-create", description: "Enter skill-authoring mode" },
  { name: "plan", description: "Show current plan state" },
  { name: "agents", description: "List registered agents" },
  { name: "permissions", description: "Change permission mode" },
  { name: "yolo", description: "Toggle accept-all permission mode" },
  { name: "thinking", description: "Toggle thinking-chunk display" },
  { name: "verbose", description: "Toggle verbose event display" },
  { name: "debug", description: "Toggle debug overlay" },
  { name: "config", description: "Show effective config" },
  { name: "history", description: "Show session history" },
  { name: "exit", description: "Quit" },
];

/** Fuzzy match: return commands whose name starts with or contains the query (case-insensitive).
 *  Dynamic skills are appended at the end of the list.
 */
export function matchCommands(query: string, skills: string[] = []): SlashCommand[] {
  const q = query.toLowerCase();
  const all = [
    ...SLASH_COMMANDS,
    ...skills.map((s) => ({ name: s, description: "Run skill" })),
  ];
  if (!q) return all;
  const prefix = all.filter((c) => c.name.toLowerCase().startsWith(q));
  const contains = all.filter(
    (c) => !c.name.toLowerCase().startsWith(q) && c.name.toLowerCase().includes(q),
  );
  return [...prefix, ...contains];
}

interface Props {
  query: string;
  selectedIndex: number;
  maxRows?: number;
  skills?: string[];
}

export const SlashMenu: React.FC<Props> = ({
  query,
  selectedIndex,
  maxRows = 8,
  skills = [],
}) => {
  const all = matchCommands(query, skills);
  if (all.length === 0) {
    return (
      <Box
        marginTop={1}
        borderStyle="round"
        borderColor="gray"
        paddingX={1}
        flexDirection="column"
      >
        <Text dimColor>no matching command</Text>
      </Box>
    );
  }

  // Sliding window around the selected index so long lists stay within maxRows.
  let start = 0;
  if (all.length > maxRows) {
    start = Math.max(0, Math.min(all.length - maxRows, selectedIndex - Math.floor(maxRows / 2)));
  }
  const visible = all.slice(start, start + maxRows);

  return (
    <Box
      marginTop={1}
      borderStyle="round"
      borderColor="magenta"
      paddingX={1}
      flexDirection="column"
    >
      {visible.map((cmd, i) => {
        const actualIndex = start + i;
        const selected = actualIndex === selectedIndex;
        return (
          <Box key={cmd.name}>
            <Text color={selected ? "magenta" : undefined} bold={selected}>
              {selected ? "› " : "  "}/{cmd.name}
            </Text>
            <Text dimColor> {cmd.description}</Text>
          </Box>
        );
      })}
      {all.length > maxRows && (
        <Text dimColor>
          {`  ${selectedIndex + 1} / ${all.length}`}
        </Text>
      )}
    </Box>
  );
};
