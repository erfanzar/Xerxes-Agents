import React, { useMemo } from "react";
import { Box, Text } from "ink";
import Gradient from "ink-gradient";
import figlet from "figlet";
import ansiShadow from "figlet/importable-fonts/ANSI Shadow.js";

figlet.parseFont("ANSI Shadow", ansiShadow);

// Persian royal duotone: imperial violet (Tyrian) → Achaemenid gold.
const PERSIAN_ROYAL = ["#9333EA", "#C084FC", "#FDE68A", "#C9A227"];

export const Banner: React.FC = () => {
  const logo = useMemo(
    () => figlet.textSync("XERXES", { font: "ANSI Shadow" }),
    [],
  );
  return (
    <Box flexDirection="column" marginY={1}>
      <Gradient colors={PERSIAN_ROYAL}>
        <Text>{logo}</Text>
      </Gradient>
    </Box>
  );
};
