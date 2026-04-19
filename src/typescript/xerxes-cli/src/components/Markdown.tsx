import React from "react";
import { Box, Text } from "ink";

const INLINE_RE = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;

function parseInline(text: string): React.ReactNode[] {
  const parts = text.split(INLINE_RE);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <Text key={i} bold>
          {part.slice(2, -2)}
        </Text>
      );
    }
    if (
      part.startsWith("*") &&
      part.endsWith("*") &&
      part.length > 1 &&
      !part.startsWith("**")
    ) {
      return (
        <Text key={i} italic>
          {part.slice(1, -1)}
        </Text>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <Text key={i} color="cyan">
          {part.slice(1, -1)}
        </Text>
      );
    }
    const linkMatch = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (linkMatch) {
      return (
        <Text key={i} color="cyan" underline>
          {linkMatch[1]}
        </Text>
      );
    }
    return part;
  });
}

export const MarkdownText: React.FC<{
  text: string;
  streaming?: boolean;
}> = ({ text, streaming }) => {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeLang = "";
  let codeLines: string[] = [];
  let keyIdx = 0;
  const nextKey = () => `md-${keyIdx++}`;

  const flushCodeBlock = () => {
    if (codeLines.length === 0) return;
    elements.push(
      <Box
        key={nextKey()}
        flexDirection="column"
        paddingLeft={2}
        paddingRight={1}
      >
        {codeLang && (
          <Text dimColor italic>
            {codeLang}
          </Text>
        )}
        {codeLines.map((l, j) => (
          <Text key={j} dimColor>
            {l}
          </Text>
        ))}
      </Box>,
    );
    codeLines = [];
    codeLang = "";
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    // Code block fences
    if (trimmed.startsWith("```")) {
      if (inCodeBlock) {
        flushCodeBlock();
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
        codeLang = trimmed.slice(3).trim();
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // Horizontal rule
    if (/^(\*{3,}|-{3,}|_{3,})$/.test(trimmed)) {
      elements.push(
        <Box key={nextKey()}>
          <Text dimColor>
            {"────────────────────────────────────────"}
          </Text>
        </Box>,
      );
      continue;
    }

    // Header
    const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      const content = headerMatch[2];
      const color = level === 1 ? "magenta" : level === 2 ? "cyan" : "white";
      elements.push(
        <Box key={nextKey()}>
          <Text bold color={color}>
            {parseInline(content)}
          </Text>
        </Box>,
      );
      continue;
    }

    // Quote
    if (line.startsWith("> ")) {
      elements.push(
        <Box key={nextKey()} paddingLeft={2}>
          <Text dimColor italic>
            ▎ {parseInline(line.slice(2))}
          </Text>
        </Box>,
      );
      continue;
    }

    // Table block (consecutive lines starting with |)
    if (trimmed.startsWith("|")) {
      const tableRows: string[] = [];
      let j = i;
      while (j < lines.length && lines[j].trim().startsWith("|")) {
        tableRows.push(lines[j]);
        j++;
      }
      i = j - 1; // skip consumed lines

      const isSeparator = (row: string) =>
        /^\|[\s|:|-]+\|$/.test(row.trim()) && !row.trim().replace(/\|/g, "").replace(/[-\s:]/g, "");

      // Strip leading/trailing pipes and split columns for cleaner display
      const stripRow = (row: string): string[] =>
        row
          .trim()
          .split("|")
          .map((c) => c.trim())
          .filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);

      // Compute max column widths for alignment
      let maxCols = 0;
      const parsedRows = tableRows.map((row) => {
        const cols = stripRow(row);
        maxCols = Math.max(maxCols, cols.length);
        return cols;
      });

      const colWidths: number[] = new Array(maxCols).fill(0);
      for (const cols of parsedRows) {
        for (let ci = 0; ci < cols.length; ci++) {
          // Strip inline markdown markers for width calculation
          const plain = cols[ci].replace(/\*\*|\*|`/g, "");
          colWidths[ci] = Math.max(colWidths[ci], plain.length);
        }
      }
      // Cap minimum width for readability
      for (let ci = 0; ci < colWidths.length; ci++) {
        colWidths[ci] = Math.max(colWidths[ci], 3);
      }

      const pad = (s: string, width: number): string => {
        const plain = s.replace(/\*\*|\*|`/g, "");
        const padLen = Math.max(0, width - plain.length);
        return s + " ".repeat(padLen);
      };

      elements.push(
        <Box key={nextKey()} flexDirection="column">
          {parsedRows.map((cols, ri) => {
            const isSep = ri === 1 && isSeparator(tableRows[ri]);
            const isHeader = ri === 0;
            // Render as padded columns separated by two spaces
            const line = cols.map((c, ci) => pad(c, colWidths[ci] ?? 3)).join("  ");
            if (isSep) {
              const sepLine = colWidths.map((w) => "─".repeat(w)).join("──");
              return (
                <Text key={ri} dimColor>
                  {sepLine}
                </Text>
              );
            }
            if (isHeader) {
              return (
                <Text key={ri} bold>
                  {line}
                </Text>
              );
            }
            return <Text key={ri}>{parseInline(line)}</Text>;
          })}
        </Box>,
      );
      continue;
    }

    // Unordered list
    const ulMatch = line.match(/^(\s*)[-*]\s+(.+)$/);
    if (ulMatch) {
      const indent = ulMatch[1].length;
      const content = ulMatch[2];
      elements.push(
        <Box key={nextKey()} paddingLeft={2 + indent}>
          <Text>• {parseInline(content)}</Text>
        </Box>,
      );
      continue;
    }

    // Ordered list
    const olMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);
    if (olMatch) {
      const indent = olMatch[1].length;
      const num = olMatch[2];
      const content = olMatch[3];
      elements.push(
        <Box key={nextKey()} paddingLeft={2 + indent}>
          <Text dimColor>{num}. </Text>
          <Text>{parseInline(content)}</Text>
        </Box>,
      );
      continue;
    }

    // Empty line → spacing
    if (!trimmed) {
      elements.push(
        <Box key={nextKey()}>
          <Text> </Text>
        </Box>,
      );
      continue;
    }

    // Regular paragraph
    elements.push(
      <Box key={nextKey()}>
        <Text>{parseInline(line)}</Text>
      </Box>,
    );
  }

  if (inCodeBlock) {
    flushCodeBlock();
  }

  // Post-process: collapse consecutive empty-line boxes and trim trailing ones
  const cleaned: React.ReactNode[] = [];
  let lastWasEmpty = false;
  for (const el of elements) {
    // Heuristic: empty-line boxes have no props other than key and a single Text child with " "
    const isEmptyBox =
      React.isValidElement(el) &&
      el.type === Box &&
      (el.props as Record<string, unknown>).children &&
      React.isValidElement((el.props as Record<string, unknown>).children) &&
      ((el.props as Record<string, unknown>).children as React.ReactElement).type === Text &&
      (((el.props as Record<string, unknown>).children as React.ReactElement).props as Record<string, unknown>).children === " ";
    if (isEmptyBox) {
      if (!lastWasEmpty) {
        cleaned.push(el);
        lastWasEmpty = true;
      }
      continue;
    }
    cleaned.push(el);
    lastWasEmpty = false;
  }
  // Trim trailing empty boxes
  while (cleaned.length > 0) {
    const last = cleaned[cleaned.length - 1];
    const isEmptyBox =
      React.isValidElement(last) &&
      last.type === Box &&
      (last.props as Record<string, unknown>).children &&
      React.isValidElement((last.props as Record<string, unknown>).children) &&
      ((last.props as Record<string, unknown>).children as React.ReactElement).type === Text &&
      (((last.props as Record<string, unknown>).children as React.ReactElement).props as Record<string, unknown>).children === " ";
    if (isEmptyBox) {
      cleaned.pop();
    } else {
      break;
    }
  }

  // Streaming cursor appended to the last element
  if (streaming) {
    cleaned.push(
      <Text key={nextKey()} dimColor>
        ▋
      </Text>,
    );
  }

  return <Box flexDirection="column">{cleaned}</Box>;
};
